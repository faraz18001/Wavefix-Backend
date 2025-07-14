from importlib import metadata
import os
import logging
import pathlib
# import json # Not strictly needed for this change, but good for other JSON tasks
from typing import Any, List, Dict, Optional as TypingOptional
from datetime import datetime, timedelta, timezone as dt_timezone
from contextlib import asynccontextmanager # For FastAPI lifespan

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Depends, status, APIRouter
from fastapi. security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr # Removed validator, not used
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import JWTError, jwt
cache_dir = "/tmp/huggingface_cache"



# Updated LangChain Imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
)
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.messages import HumanMessage, AIMessage

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_app.db")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-please-change")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4" # Main LLM for RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
# SESSION_NAME_MAX_LENGTH = 10 # Removed, LLM will generate title with its own length constraints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Password Hashing & JWT ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

class TokenData(BaseModel):
    username: TypingOptional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: TypingOptional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(dt_timezone.utc) + expires_delta
    else:
        expire = datetime.now(dt_timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False) # "user" or "ai"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")

# --- Pydantic Schemas (API Models) ---
class UserCreate(BaseModel):
    username: str
    email: TypingOptional[EmailStr] = None
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: TypingOptional[EmailStr] = None
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str
    session_id: TypingOptional[str] = None

class SourceDocumentModel(BaseModel):
    document_name:str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocumentModel]
    session_id: str
    tokens_used: int
    cost_usd: float

class ChatSessionInfo(BaseModel):
    id: int
    name: TypingOptional[str] = "New Chat"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

# <<<< ----- NEW: Pydantic model for simple API responses ----- >>>>
class DetailResponse(BaseModel):
    detail: str


# --- CRUD Operations ---
def get_user_by_username(db: Session, username: str) -> TypingOptional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> TypingOptional[User]:
    return db.query(User).filter(User.email == email).first()

def create_db_user(db: Session, user: UserCreate) -> User:
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_or_create_chat_session(db: Session, user_id: int, client_provided_session_id: TypingOptional[str] = None) -> ChatSession:
    if client_provided_session_id:
        try:
            session_db_id = int(client_provided_session_id)
            chat_session = db.query(ChatSession).filter(ChatSession.user_id == user_id, ChatSession.id == session_db_id).first()
            if chat_session:
                logger.info(f"Resuming chat session {session_db_id} (name: '{chat_session.name}') for user {user_id}")
                chat_session.updated_at = datetime.now(dt_timezone.utc)
                db.commit()
                db.refresh(chat_session)
                return chat_session
            else:
                logger.info(f"Session ID {client_provided_session_id} not found for user {user_id}. Creating new.")
        except ValueError:
            logger.warning(f"Invalid session_id format: {client_provided_session_id}. Creating new session.")
    
    new_chat_session = ChatSession(user_id=user_id, name=None) 
    db.add(new_chat_session)
    db.commit()
    db.refresh(new_chat_session)
    logger.info(f"Created new chat session {new_chat_session.id} for user {user_id} (name pending).")
    return new_chat_session

def get_chat_history_from_db(db: Session, session_id: int) -> ChatMessageHistory:
    db_messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
    history = ChatMessageHistory()
    for msg in db_messages:
        if msg.role == "user":
            history.add_message(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            history.add_message(AIMessage(content=msg.content))
    logger.debug(f"Retrieved {len(history.messages)} messages for session {session_id} from DB for LangChain history.")
    return history

def add_chat_message_to_db(db: Session, session_id: int, role: str, content: str, sources: TypingOptional[List[SourceDocumentModel]] = None) -> ChatMessage:
    db_message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(db_message)
    parent_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if parent_session:
        parent_session.updated_at = datetime.now(dt_timezone.utc)
    db.commit() 
    db.refresh(db_message)
    logger.debug(f"Saved '{role}' message to DB for session {session_id}.")
    return db_message

# --- User Authentication Dependencies ---
async def get_current_db_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        logger.warning(f"JWTError while decoding token: {token[:20]}...")
        raise credentials_exception
    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_db_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Document Loading Logic ---
class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | List[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass

class DocumentLoader:
    supported_extensions = {
        ".pdf": PyPDFLoader, ".txt": TextLoader, ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
    }
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        ext = pathlib.Path(file_path).suffix.lower()
        loader_class = DocumentLoader.supported_extensions.get(ext)
        if not loader_class:
            raise DocumentLoaderException(
                f"Unsupported file extension: {ext}. Supported are: {', '.join(DocumentLoader.supported_extensions.keys())}"
            )
        logger.info(f"Loading document: {file_path} using {loader_class.__name__}")
        loader = loader_class(file_path)
        try:
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
            return docs
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise DocumentLoaderException(f"Could not load {file_path}: {e}")

    @staticmethod
    def load_documents_from_directory(directory_path: str) -> List[Document]:
        all_docs: List[Document] = []
        abs_directory_path = pathlib.Path(directory_path).resolve()
        if not abs_directory_path.is_dir():
            logger.warning(f"Directory not found: {abs_directory_path}")
            return []
        logger.info(f"Scanning for documents in: {abs_directory_path}")
        found_files = False
        for file_path in abs_directory_path.rglob("*"):
            if file_path.is_file():
                found_files = True
                logger.debug(f"Found file: {file_path.name}")
                try:
                    docs = DocumentLoader.load_document(str(file_path))
                    all_docs.extend(docs)
                except DocumentLoaderException as e:
                    logger.warning(f"Skipping file {file_path.name}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing file {file_path.name}: {e}")
        if not found_files: logger.warning(f"No files found in directory: {abs_directory_path}")
        elif not all_docs: logger.warning(f"No processable documents found in {abs_directory_path}.")
        logger.info(f"Total documents loaded from directory: {len(all_docs)}")
        return all_docs

# --- Retriever Configuration ---
def configure_retriever(docs: List[Document]) -> TypingOptional[BaseRetriever]:
    if not docs:
        logger.warning("No documents for retriever.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    if not splits:
        logger.warning("No text splits generated. Retriever will not be functional.")
        return None
    logger.info(f"Created {len(splits)} text splits.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(splits, embeddings)
    logger.info("FAISS vector store created.")
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

# --- Prompt Creation ---
def create_custom_prompt():
    system_template = """You are a friendly and expert technical support agent.
Your primary job is to provide clear solutions to user problems based ONLY on the provided context from the knowledge base, user questions, and chat history.
Always provide practical, actionable advice.
You can engage in brief, polite greetings or acknowledgments. For example, if the user says 'hello' or 'hi', you can respond with a friendly greeting like 'Hello! How can I assist you with your technical product questions today?' or 'Hi there! What technical issue can I help you with?', and then wait for their actual question. Do not try to find greetings in the knowledge base.
For technical questions:
- If the user's question is not clear or you need more information, ask for clarification.
- If the solution is not evident from the context, state that you cannot find the information in the provided knowledge base. DO NOT use external knowledge for technical answers.
- Provide up to 4 most relevant probable solutions based on the context.
- Each solution should ideally be concise.
- ONLY answer technical questions based on the provided context (knowledge base).
- If the user asks a technical question that is unrelated to the context or something you cannot find in the context, respond with: "I'm sorry, I can only assist with technical questions related to the provided knowledge base, and I could not find information on that specific topic. Is there a different technical issue I can help you with?"
Remember for technical answers:
- Don't give step by step solution (unless the context explicitly provides it as the best format).
- Just provide up to 4 most closest probable solutions.
- Each solution must be in one line if possible, or a very short paragraph.

Chat History:
{chat_history}
"""
    human_template = """Context from Knowledge Base:
{context}

User Question: {question}

Please provide your recommendation:"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# --- Global Variables for Langchain Components ---
retriever_global: TypingOptional[BaseRetriever] = None
llm_global: TypingOptional[ChatOpenAI] = None
prompt_global: TypingOptional[ChatPromptTemplate] = None
llm_chain_global: TypingOptional[RunnableSequence] = None
output_parser_global: TypingOptional[StrOutputParser] = None

# --- FastAPI Application Lifespan ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global retriever_global, llm_global, prompt_global, llm_chain_global, output_parser_global
    logger.info("Application startup: Initializing DB, loading documents, RAG components...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")

    docs = DocumentLoader.load_documents_from_directory(DOCUMENTS_DIR)
    if not docs: logger.warning(f"No documents loaded from {DOCUMENTS_DIR}. RAG system might be ineffective.")
    
    retriever_global = configure_retriever(docs)
    if retriever_global is None: logger.warning("Retriever not configured. RAG system might be ineffective.")
    
    # <<<<<<<<<<<<<<<< THIS IS THE CORRECTED LINE >>>>>>>>>>>>>>>>
    llm_global = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL_NAME, temperature=0.3)
    
    prompt_global = create_custom_prompt()
    output_parser_global = StrOutputParser()

    if llm_global and prompt_global and output_parser_global:
        llm_chain_global = prompt_global | llm_global | output_parser_global
        logger.info("LLM RunnableSequence initialized.")
    else:
        logger.error("LLM, Prompt or OutputParser not available, RunnableSequence not initialized.")
    
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")

# --- FastAPI Application ---
app = FastAPI(
    title="Tech Support Bot API with Auth",
    description="API for a RAG tech support bot with user authentication, persistent chat history, and dynamic session naming.",
    version="1.5.0", # Incremented version for delete session functionality
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://13.58.122.221:7000", "http://localhost:3000", "http://13.58.122.221:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Router ---
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user_by_username = get_user_by_username(db, username=user.username)
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Username already registered")
    if user.email:
        db_user_by_email = get_user_by_email(db, email=user.email)
        if db_user_by_email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    logger.info(f"Attempting to create user: {user.username}")
    try:
        created_user = create_db_user(db=db, user=user)
        logger.info(f"User '{created_user.username}' created successfully with id {created_user.id}")
        return created_user
    except Exception as e:
        logger.error(f"Error during user creation for {user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during user creation: {str(e)}")

@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

app.include_router(auth_router)

# --- Chat Router ---
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

# --- LLM-BASED SESSION NAME GENERATION LOGIC ---
def _fallback_generate_name_v2(query: str, max_words: int = 5) -> str:
    """Generates a simple fallback session name from the query."""
    if not query or not query.strip():
        return "Chat"
    
    words = query.strip().split()
    if not words:
        return "Chat"
    
    fallback_title = " ".join(words[:max_words])
    if len(words) > max_words:
        fallback_title += "..."
    
    return fallback_title if fallback_title else "Chat"

async def generate_session_name_llm(user_query: str) -> str:
    """Generates a session name using an LLM, with a fallback."""
    if not user_query or not user_query.strip():
        return "Chat" # Default for empty query, avoid LLM call

    system_template_str = """You are an AI assistant. Your task is to create a concise title for a user's message.
The title must be 2-6 words long.
It should capture the main topic or essence of the user's message.
Use clear, simple language.
Avoid filler words (e.g., "about", "regarding").
Do not include prefixes like "Title:". Output only the title itself.
For example, if the user message is "My computer won't turn on after the power outage", a good title is "Computer Fails to Start".
If the user message is "Hi there, how are you?", a good title is "General Greeting".
"""
    human_template_str = "{user_query_for_titleing}"

    system_prompt_obj = SystemMessagePromptTemplate.from_template(system_template_str)
    human_prompt_obj = HumanMessagePromptTemplate.from_template(human_template_str)
    chat_prompt_for_title = ChatPromptTemplate.from_messages([system_prompt_obj, human_prompt_obj])
    
    title_llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1, 
        max_tokens=25      
    )

    try:
        formatted_messages = chat_prompt_for_title.format_messages(user_query_for_titleing=user_query)
        response = await title_llm.ainvoke(formatted_messages)
        title = response.content.strip().strip('"').strip("'") 

        if not title: 
            logger.warning(f"LLM generated an empty title for query: '{user_query[:50]}...'. Falling back.")
            return _fallback_generate_name_v2(user_query)

        words = title.split()
        if not (2 <= len(words) <= 7):
            logger.warning(f"LLM title '{title}' (words: {len(words)}) for query '{user_query[:50]}...' is outside 2-6 word target. Using fallback.")
            return _fallback_generate_name_v2(user_query)

        logger.info(f"LLM generated title: '{title}' for query: '{user_query[:50]}...'")
        return title

    except Exception as e:
        logger.error(f"Error generating session name with LLM for query '{user_query[:50]}...': {e}", exc_info=True)
        return _fallback_generate_name_v2(user_query)
@chat_router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    global retriever_global, llm_chain_global

    logger.info(
        f"Chat request for user '{current_user.username}', client session_id: '{request.session_id}', query: '{request.query[:50]}...'"
    )

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if retriever_global is None or llm_chain_global is None:
        logger.error("RAG system not initialized. Cannot process request.")
        raise HTTPException(status_code=503, detail="Service not fully initialized.")

    db_chat_session = get_or_create_chat_session(db, user_id=current_user.id, client_provided_session_id=request.session_id)
    session_id_for_response = str(db_chat_session.id)
    
    should_generate_name = db_chat_session.name is None
    current_langchain_history = get_chat_history_from_db(db, session_id=db_chat_session.id)

    try:
        with get_openai_callback() as cb:
            relevant_docs = await retriever_global.aget_relevant_documents(request.query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            chat_history_str = "\n".join(
                [f"{msg.type.capitalize()}: {msg.content}" for msg in current_langchain_history.messages]
            )
            inputs = {
                "context": context or "No relevant context found in the knowledge base.",
                "question": request.query,
                "chat_history": chat_history_str,
            }
            result_str = await llm_chain_global.ainvoke(inputs)
            
            source_docs_model = [
                SourceDocumentModel(document_name=os.path.basename(doc.metadata.get('source', 'Unknown Document')))
                for doc in relevant_docs
            ]

            add_chat_message_to_db(db, session_id=db_chat_session.id, role="user", content=request.query)
            add_chat_message_to_db(db, session_id=db_chat_session.id, role="ai", content=result_str, sources=source_docs_model)

            if should_generate_name:
                generated_name = await generate_session_name_llm(request.query)
                current_db_session_for_name = db.query(ChatSession).filter(ChatSession.id == db_chat_session.id).first()
                if current_db_session_for_name:
                    current_db_session_for_name.name = generated_name
                    db.commit()
                    logger.info(f"Set session {current_db_session_for_name.id} name to '{generated_name}' for user '{current_user.username}' using LLM.")
                else:
                    logger.error(f"Session {db_chat_session.id} not found when trying to set LLM-generated name.")

            logger.info(
                f"Response for user '{current_user.username}', session {session_id_for_response}. Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}"
            )
            return ChatResponse(
                answer=result_str,
                sources=source_docs_model,
                session_id=session_id_for_response,
                tokens_used=cb.total_tokens,
                cost_usd=cb.total_cost,
            )
    except Exception as e:
        logger.error(f"Error processing chat for user {current_user.username}, session {session_id_for_response}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@chat_router.get("/sessions", response_model=List[ChatSessionInfo])
async def list_user_chat_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [ChatSessionInfo.from_orm(s) for s in sessions]


@chat_router.get("/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    chat_session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")

    db_messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
    
    session_display_name = chat_session.name if chat_session.name else "New Chat"
    logger.debug(f"Retrieved {len(db_messages)} messages for session {session_id} (name: '{session_display_name}') for user {current_user.username}")
    return [ChatMessageResponse.from_orm(msg) for msg in db_messages]

# <<<< ----- NEW: DELETE endpoint to clear a chat session ----- >>>>
@chat_router.delete(
    "/sessions/{session_id}",
    response_model=DetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a chat session"
)
async def delete_chat_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Deletes a specific chat session and all of its associated messages.
    The user must be the owner of the chat session to delete it.
    """
    logger.info(f"User '{current_user.username}' attempting to delete session {session_id}")

    # Find the session, ensuring it belongs to the current user for security
    chat_session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not chat_session:
        logger.warning(
            f"Failed delete attempt: Session {session_id} not found or user '{current_user.username}' does not have access."
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied")

    try:
        # Delete the session. The `cascade="all, delete-orphan"` in the model
        # will automatically delete all related ChatMessage records.
        db.delete(chat_session)
        db.commit()
        logger.info(f"Successfully deleted chat session {session_id} for user '{current_user.username}'.")
        return {"detail": "Chat session deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Database error while deleting session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete chat session due to a server error."
        )

app.include_router(chat_router)

# --- Health Check ---
@app.get("/health")
async def health_check():
    db_healthy = False
    try:
        db = SessionLocal()
        db.execute(func.text("SELECT 1")) 
        db_healthy = True
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
    finally:
        if 'db' in locals() and db: 
            db.close()

    rag_components_ok = retriever_global and llm_chain_global
    
    if db_healthy and rag_components_ok:
        return {"status": "ok", "message": "Service is operational."}
    
    details = []
    if not db_healthy: details.append("Database connectivity issue.")
    if not rag_components_ok: details.append("RAG components (retriever or LLM chain) not fully initialized.")
    
    return {"status": "degraded", "message": "Service issues detected.", "details": details}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")

    db_path_str = DATABASE_URL.split("sqlite:///")[1] if "sqlite:///" in DATABASE_URL else None
    if db_path_str:
        db_file = pathlib.Path(db_path_str)
        if not db_file.exists():
            logger.info(f"SQLite database not found at {db_file}, it will be created by SQLAlchemy.")
        else:
            logger.info(f"Using existing SQLite database at {db_file}. "
                        "If schema changes were made, "
                        "you may need to delete this file for changes to apply with SQLite, "
                        "or use a migration tool for production.")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
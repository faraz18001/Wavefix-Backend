from importlib import metadata
import os
import logging
import pathlib
from typing import Any, List, Dict, Optional as TypingOptional, TypedDict

from datetime import datetime, timedelta, timezone as dt_timezone
from contextlib import asynccontextmanager # For FastAPI lifespan

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Depends, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import JWTError, jwt

# LangChain and LangGraph Imports
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
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage # Added BaseMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # For persistent graph state (optional here, using DB directly)

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_app_langgraph.db") # Changed DB name slightly
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-please-change")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4"
TITLE_LLM_MODEL_NAME = "gpt-3.5-turbo" # For session titles
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 1 # Keep this low to stay within prompt limits, can be adjusted

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

# --- Database Models (Identical to original) ---
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
    name = Column(String(255), nullable=True, index=True) # Name can be null initially
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

# --- Pydantic Schemas (API Models - Identical to original) ---
class UserCreate(BaseModel):
    username: str
    email: TypingOptional[EmailStr] = None
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: TypingOptional[EmailStr] = None
    is_active: bool
    class Config: from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str
    session_id: TypingOptional[str] = None # Client can send string, we convert to int for DB

class SourceDocumentModel(BaseModel):
    document_name: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocumentModel]
    session_id: str # Return actual DB session ID as string
    tokens_used: int
    cost_usd: float
    generated_session_name: TypingOptional[str] = None # If a new name was generated

class ChatSessionInfo(BaseModel):
    id: int
    name: TypingOptional[str] = "New Chat"
    created_at: datetime
    updated_at: datetime
    class Config: from_attributes = True

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    timestamp: datetime
    class Config: from_attributes = True

# --- CRUD Operations (Mostly identical, minor adjustments for session name) ---
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
                chat_session.updated_at = datetime.now(dt_timezone.utc) # Touch session on resume
                db.commit()
                db.refresh(chat_session)
                return chat_session
            else:
                logger.info(f"Session ID {client_provided_session_id} not found for user {user_id}. Creating new.")
        except ValueError:
            logger.warning(f"Invalid session_id format: {client_provided_session_id}. Creating new session.")
    
    # Create new session without a name initially
    new_chat_session = ChatSession(user_id=user_id, name=None) 
    db.add(new_chat_session)
    db.commit()
    db.refresh(new_chat_session)
    logger.info(f"Created new chat session {new_chat_session.id} for user {user_id}.")
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

def add_chat_message_to_db(db: Session, session_id: int, role: str, content: str) -> ChatMessage:
    # Sources are not directly stored with the message in this simplified model.
    # They are derived from the retriever step.
    db_message = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(db_message)
    
    # Touch the parent session to update its updated_at timestamp
    parent_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if parent_session:
        parent_session.updated_at = datetime.now(dt_timezone.utc)
    
    db.commit()
    db.refresh(db_message)
    logger.debug(f"Saved '{role}' message to DB for session {session_id}.")
    return db_message

def update_session_name_in_db(db: Session, session_id: int, name: str) -> None:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.name = name
        session.updated_at = datetime.now(dt_timezone.utc) # Also update timestamp
        db.commit()
        logger.info(f"Updated session {session_id} name to '{name}'.")
    else:
        logger.warning(f"Could not find session {session_id} to update its name.")


# --- User Authentication Dependencies (Identical to original) ---
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

# --- Document Loading Logic (Identical to original) ---
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

# --- Retriever Configuration (Identical to original) ---
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

# --- Prompt Creation (Identical to original) ---
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

# --- Session Name Generation Logic (Modified for graph node) ---
def _fallback_generate_name_internal(query: str, max_words: int = 5) -> str:
    if not query or not query.strip(): return "Chat"
    words = query.strip().split()
    if not words: return "Chat"
    fallback_title = " ".join(words[:max_words])
    if len(words) > max_words: fallback_title += "..."
    return (fallback_title if fallback_title else "Chat")[:250]

async def generate_session_name_llm_internal(user_query: str, title_llm: ChatOpenAI) -> str:
    if not user_query or not user_query.strip(): return "Chat"
    
    system_template_str = """You are an AI assistant. Your task is to create a concise title for a user's message.
The title must be 2-6 words long. It should capture the main topic or essence of the user's message.
Use clear, simple language. Avoid filler words. Do not include prefixes like "Title:". Output only the title itself.
E.g., User: "My computer won't turn on after the power outage", Title: "Computer Fails to Start".
User: "Hi there, how are you?", Title: "General Greeting".
User: "test", Title: "Test Inquiry".
"""
    human_template_str = "{user_query_for_titleing}"
    system_prompt_obj = SystemMessagePromptTemplate.from_template(system_template_str)
    human_prompt_obj = HumanMessagePromptTemplate.from_template(human_template_str)
    chat_prompt_for_title = ChatPromptTemplate.from_messages([system_prompt_obj, human_prompt_obj])
    
    try:
        formatted_messages = chat_prompt_for_title.format_messages(user_query_for_titleing=user_query)
        response = await title_llm.ainvoke(formatted_messages)
        title = response.content.strip().strip('"').strip("'")
        if not title:
            logger.warning(f"LLM generated empty title for '{user_query[:50]}...'. Falling back.")
            return _fallback_generate_name_internal(user_query)
        words = title.split()
        if not (1 <= len(words) <= 7):
            logger.warning(f"LLM title '{title}' ({len(words)} words) for '{user_query[:50]}...' outside 1-7 words. Falling back.")
            return _fallback_generate_name_internal(user_query)
        logger.info(f"LLM generated title: '{title}' for query: '{user_query[:50]}...'")
        return title[:250]
    except Exception as e:
        logger.error(f"LLM error for title gen on '{user_query[:50]}...': {e}", exc_info=True)
        return _fallback_generate_name_internal(user_query)

# --- LangGraph State Definition ---
class ChatGraphState(TypedDict):
    user_id: int
    db_session_id: int # The actual ID from the database
    user_query: str
    chat_history_langchain: ChatMessageHistory # LangChain message objects
    
    # For RAG
    context: TypingOptional[str]
    sources: List[SourceDocumentModel]
    
    # LLM Output
    answer: TypingOptional[str]
    
    # For session naming
    should_generate_name: bool # Flag to indicate if name generation is needed
    generated_session_name: TypingOptional[str]
    
    # Cost tracking
    tokens_used: int
    cost_usd: float
    
    # To pass DB session to nodes
    # This is not serializable by default for SqliteSaver, but we are not using SqliteSaver persistence here.
    # If you were, you'd manage DB sessions outside or pass only IDs.
    db: Session 


# --- LangGraph Nodes ---
async def retrieve_documents_node(state: ChatGraphState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(f"Graph Node: retrieve_documents_node for session {state['db_session_id']}")
    query = state["user_query"]
    if not retriever_global:
        logger.warning("Retriever not available in retrieve_documents_node.")
        return {"context": "Retriever not configured.", "sources": []}
        
    relevant_docs = await retriever_global.aget_relevant_documents(query, config=config)
    context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
    source_docs_model = [
        SourceDocumentModel(document_name=os.path.basename(doc.metadata.get('source', 'Unknown Document')))
        for doc in relevant_docs
    ]
    return {"context": context_str or "No relevant context found.", "sources": source_docs_model}

async def generate_response_node(state: ChatGraphState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(f"Graph Node: generate_response_node for session {state['db_session_id']}")
    if not llm_global or not rag_prompt_global or not output_parser_global:
        logger.error("LLM, Prompt or OutputParser not available for RAG.")
        return {"answer": "RAG system components are not initialized."}

    chat_history_str = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in state["chat_history_langchain"].messages[-20:]]
    )
    inputs = {
        "context": state["context"],
        "question": state["user_query"],
        "chat_history": chat_history_str,
    }
    
    chain = rag_prompt_global | llm_global | output_parser_global
    
    tokens_used = 0
    cost_usd = 0.0
    
    with get_openai_callback() as cb:
        answer = await chain.ainvoke(inputs, config=config)
        tokens_used = cb.total_tokens
        cost_usd = cb.total_cost
        
    logger.info(f"Generated response for session {state['db_session_id']}. Tokens: {tokens_used}, Cost: ${cost_usd:.6f}")
    return {"answer": answer, "tokens_used": tokens_used, "cost_usd": cost_usd}

async def save_chat_history_node(state: ChatGraphState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(f"Graph Node: save_chat_history_node for session {state['db_session_id']}")
    db = state["db"] # Get DB session from state
    
    # Save user message
    add_chat_message_to_db(db, session_id=state["db_session_id"], role="user", content=state["user_query"])
    
    # Save AI message
    if state.get("answer"):
        add_chat_message_to_db(db, session_id=state["db_session_id"], role="ai", content=state["answer"])
    else:
        logger.warning("No AI answer found in state to save.")
        
    # No state change needed to return, side effect is DB write
    return {} 

async def generate_session_name_node(state: ChatGraphState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(f"Graph Node: generate_session_name_node for session {state['db_session_id']}")
    db = state["db"]
    user_query = state["user_query"]
    
    if not title_llm_global:
        logger.warning("Title LLM not available. Using fallback name.")
        generated_name = _fallback_generate_name_internal(user_query)
    else:
        generated_name = await generate_session_name_llm_internal(user_query, title_llm_global)
    
    update_session_name_in_db(db, state["db_session_id"], generated_name)
    return {"generated_session_name": generated_name}

# --- LangGraph Conditional Edges ---
def should_generate_name_edge(state: ChatGraphState) -> str:
    if state.get("should_generate_name", False): # Check the flag set initially
        logger.info(f"Conditional Edge: Routing to generate_session_name for session {state['db_session_id']}")
        return "generate_name_needed"
    else:
        logger.info(f"Conditional Edge: Skipping name generation for session {state['db_session_id']}")
        return "generate_name_not_needed"

# --- Global Variables for Langchain/LangGraph Components ---
retriever_global: TypingOptional[BaseRetriever] = None
llm_global: TypingOptional[ChatOpenAI] = None # For RAG
title_llm_global: TypingOptional[ChatOpenAI] = None # For session titles
rag_prompt_global: TypingOptional[ChatPromptTemplate] = None
output_parser_global: TypingOptional[StrOutputParser] = None
chat_graph_global: TypingOptional[StateGraph] = None


# --- FastAPI Application Lifespan ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global retriever_global, llm_global, title_llm_global, rag_prompt_global, output_parser_global, chat_graph_global
    logger.info("Application startup: Initializing DB, loading documents, RAG components, LangGraph...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")

    docs = DocumentLoader.load_documents_from_directory(DOCUMENTS_DIR)
    if not docs: logger.warning(f"No documents loaded from {DOCUMENTS_DIR}. RAG system might be ineffective.")
    
    retriever_global = configure_retriever(docs)
    if retriever_global is None: logger.warning("Retriever not configured.")
    
    llm_global = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.3, openai_api_key=OPENAI_API_KEY)
    title_llm_global = ChatOpenAI(model_name=TITLE_LLM_MODEL_NAME, temperature=0.1, max_tokens=25, openai_api_key=OPENAI_API_KEY)
    rag_prompt_global = create_custom_prompt()
    output_parser_global = StrOutputParser()

    # Build LangGraph
    workflow = StateGraph(ChatGraphState)
    workflow.add_node("retrieve_docs", retrieve_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("save_history", save_chat_history_node)
    workflow.add_node("generate_name", generate_session_name_node)

    workflow.set_entry_point("retrieve_docs")
    workflow.add_edge("retrieve_docs", "generate_response")
    workflow.add_edge("generate_response", "save_history")
    
    # Conditional edge for session naming
    workflow.add_conditional_edges(
        "save_history", # After saving, decide if name generation is needed
        should_generate_name_edge,
        {
            "generate_name_needed": "generate_name",
            "generate_name_not_needed": END # If no name needed, end the graph here.
        }
    )
    workflow.add_edge("generate_name", END) # After generating name, end the graph.

    # memory = SqliteSaver(conn=sqlite3.connect("langgraph_memory.db", check_same_thread=False)) # Example if using memory
    # chat_graph_global = workflow.compile(checkpointer=memory) # With checkpointer
    chat_graph_global = workflow.compile() # Without checkpointer for this example
    logger.info("LangGraph compiled.")
    
    if not all([llm_global, rag_prompt_global, output_parser_global, title_llm_global, chat_graph_global]):
        logger.error("One or more critical components (LLMs, Prompt, Parser, Graph) failed to initialize.")
    else:
        logger.info("All Langchain/LangGraph components initialized.")

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")

# --- FastAPI Application ---
app = FastAPI(
    title="Tech Support Bot API with Auth (LangGraph)",
    description="API for a RAG tech support bot with user auth, persistent history, dynamic session naming, using LangGraph.",
    version="2.0.0",
    lifespan=lifespan
)

# --- CORS Middleware (Identical to original) ---
origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",") # Added 3001 for potential other frontend
logger.info(f"CORS allowed origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Router (Identical to original) ---
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active: raise HTTPException(status_code=400, detail="Inactive user")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
app.include_router(auth_router)

# --- Chat Router (Modified for LangGraph) ---
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    logger.info(f"Chat request for user '{current_user.username}', client session_id: '{request.session_id}', query: '{request.query[:50]}...'")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not chat_graph_global or not retriever_global: # Check critical components
        logger.error("Chat graph or retriever not initialized. Cannot process request.")
        raise HTTPException(status_code=503, detail="Service not fully initialized.")

    db_chat_session = get_or_create_chat_session(db, user_id=current_user.id, client_provided_session_id=request.session_id)
    session_id_for_response = str(db_chat_session.id) # Use actual DB ID

    # Flag for name generation: if it's a new session (name is None)
    # The graph will use this flag.
    needs_name_generation = db_chat_session.name is None 
    
    current_langchain_history = get_chat_history_from_db(db, session_id=db_chat_session.id)

    # Prepare initial state for the graph
    initial_state: ChatGraphState = {
        "user_id": current_user.id,
        "db_session_id": db_chat_session.id,
        "user_query": request.query,
        "chat_history_langchain": current_langchain_history,
        "context": None, # Will be populated by a node
        "sources": [],   # Will be populated by a node
        "answer": None,  # Will be populated by a node
        "should_generate_name": needs_name_generation,
        "generated_session_name": None, # Will be populated if name is generated
        "tokens_used": 0,
        "cost_usd": 0.0,
        "db": db # Pass the db session
    }
    
    # For LangGraph, thread_id is usually associated with checkpointers for persistence.
    # Since we are managing persistence via our DB, we don't strictly need a checkpointer's thread_id.
    # However, `ainvoke` expects a `configurable` dict, which often includes `thread_id`.
    # We can use our session ID as a conceptual thread_id if needed, or just an empty dict if not using checkpointer.
    # RunnableConfig can also be used to pass things like callbacks.
    run_config = RunnableConfig(
        # callbacks=[...], # if you have any LangChain callbacks
        # configurable={"thread_id": session_id_for_response} # If using checkpointer
    )

    try:
        # Invoke the graph. The `db` session is passed within `initial_state`.
        # If you had other non-serializable objects needed by nodes and not in state,
        # `config={"configurable": {"my_object": my_obj_instance}}` could be used.
        final_state = await chat_graph_global.ainvoke(initial_state, config=run_config)
        
        logger.info(
            f"Graph execution complete for session {session_id_for_response}. "
            f"Answer: {final_state['answer'][:50]}..., "
            f"Tokens: {final_state['tokens_used']}, Cost: ${final_state['cost_usd']:.6f}"
        )
        
        if final_state.get("generated_session_name"):
            logger.info(f"Session {session_id_for_response} name was set to '{final_state['generated_session_name']}'")

        return ChatResponse(
            answer=final_state.get("answer", "Error: No answer generated."),
            sources=final_state.get("sources", []),
            session_id=session_id_for_response,
            tokens_used=final_state.get("tokens_used", 0),
            cost_usd=final_state.get("cost_usd", 0.0),
            generated_session_name=final_state.get("generated_session_name")
        )

    except Exception as e:
        logger.error(f"Error processing chat graph for user {current_user.username}, session {session_id_for_response}: {e}", exc_info=True)
        # Check if it's an HTTPException from a node, otherwise wrap
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal error occurred during chat processing: {str(e)}")

# --- Other Chat Router Endpoints (Identical to original as they don't use RAG logic directly) ---
@chat_router.get("/sessions", response_model=List[ChatSessionInfo])
async def list_user_chat_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    sessions = (db.query(ChatSession).filter(ChatSession.user_id == current_user.id)
                .order_by(ChatSession.updated_at.desc()).all())
    return [ChatSessionInfo.from_orm(s) for s in sessions]

@chat_router.get("/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    chat_session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")
    db_messages = chat_session.messages
    session_display_name = chat_session.name if chat_session.name else "New Chat"
    logger.debug(f"Retrieved {len(db_messages)} messages for session {session_id} (name: '{session_display_name}') for user {current_user.username}")
    return [ChatMessageResponse.from_orm(msg) for msg in db_messages]

@chat_router.delete("/sessions", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_user_chat_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    logger.info(f"Attempting to delete all chat sessions for user '{current_user.username}' (ID: {current_user.id})")
    sessions_to_delete = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()
    if not sessions_to_delete:
        logger.info(f"No chat sessions found for user '{current_user.username}' to delete.")
        return
    num_sessions = len(sessions_to_delete)
    try:
        for session in sessions_to_delete:
            db.delete(session)
        db.commit()
        logger.info(f"Successfully deleted {num_sessions} chat session(s) for user '{current_user.username}'.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting chat sessions for user '{current_user.username}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete chat sessions.")

app.include_router(chat_router)

# --- Health Check (Identical to original, but check chat_graph_global) ---
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

    rag_components_ok = retriever_global and llm_global and title_llm_global and chat_graph_global
    
    if db_healthy and rag_components_ok:
        return {"status": "ok", "message": "Service is operational."}
    
    details = []
    if not db_healthy: details.append("Database connectivity issue.")
    if not rag_components_ok: details.append("RAG, Title LLM, or Graph components not fully initialized.")
    
    return {"status": "degraded", "message": "Service issues detected.", "details": details}

# --- Main Execution (Identical to original) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    db_path_str = DATABASE_URL.split("sqlite:///")[1] if "sqlite:///" in DATABASE_URL else None
    if db_path_str:
        db_file = pathlib.Path(db_path_str)
        if not db_file.exists():
            logger.info(f"SQLite database not found at {db_file}, it will be created by SQLAlchemy.")
        else:
            logger.info(f"Using existing SQLite database at {db_file}. Schema changes are handled by SQLAlchemy Base.metadata.create_all.")
    
    # Using "main_langgraph:app" if you save this as main_langgraph.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
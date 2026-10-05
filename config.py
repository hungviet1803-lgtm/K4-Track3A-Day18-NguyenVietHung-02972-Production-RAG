"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- LLM provider (generation, M5 enrichment, RAGAS judge) ---
# Ưu tiên OpenAI nếu có key; không thì dùng Gemini qua endpoint tương thích OpenAI
# → cùng client `openai` / `langchain_openai`, chỉ đổi base_url + model.
if OPENAI_API_KEY:
    LLM_PROVIDER = "openai"
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = None
    LLM_MODEL = "gpt-4o-mini"
    LLM_EMBEDDING_MODEL = "text-embedding-3-small"
elif GEMINI_API_KEY:
    LLM_PROVIDER = "gemini"
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    LLM_EMBEDDING_MODEL = "gemini-embedding-001"
else:
    LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_EMBEDDING_MODEL = None, "", None, "", ""

# Free tier giới hạn request/phút → client tự retry với backoff khi gặp 429
LLM_MAX_RETRIES = 8


def make_llm_client():
    """OpenAI-compatible client cho provider đang cấu hình."""
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=LLM_MAX_RETRIES)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

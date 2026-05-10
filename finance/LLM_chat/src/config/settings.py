import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

from typing import Dict

load_dotenv()

class LogSettings(BaseModel):
    LOG_LEVEL: str = "INFO"

class PathsSettings(BaseModel):
    data_dir: Dict[str, str]
    persist_dir: str
    pdf_dir: Dict[str, str]
    output_dir: str

class ProcessingSettings(BaseModel):
    chunk_size: int
    overlap: int

class RuntimeSettings(BaseModel):
    device: str

class EmbeddingsSettings(BaseModel):
    model: str
    base_url: str

class LLMModelsSettings(BaseModel):
    chat: str

class LLMParamsSettings(BaseModel):
    temperature: float
    max_new_tokens: int
    num_ctx: int
    top_k: int
    top_p: float
    seed: int = 42

class LLMSettings(BaseModel):
    provider: str = "ollama"       # <- novo campo
    base_url: str | None = None    # usado apenas se provider == ollama
    api_key: str | None = None     # usado apenas se provider == groq
    models: LLMModelsSettings
    params: LLMParamsSettings
    provider_params: Dict[str, Dict] = {}

class RerankerSettings(BaseModel):
    hf_model_path: str
    device: str
    batch_size: int
    normalize: bool

class RetrievalSettings(BaseModel):
    top_k: int
    fetch_k: int
    enable_reranker: bool

class VectorStoreSettings(BaseModel):
    chroma_path: str = "chroma_db"
    
class AppSettings(BaseModel):
    logging: LogSettings
    paths: PathsSettings
    processing: ProcessingSettings
    runtime: RuntimeSettings
    embedding: EmbeddingsSettings
    vectorstore: VectorStoreSettings
    llm: LLMSettings
    reranker: RerankerSettings
    retrieval: RetrievalSettings

def load_settings() -> AppSettings:
    env = os.getenv("APP_ENV", "dev")
    cfg_path = f"config/{env}.yaml"

    if not os.path.exists(cfg_path) and env == "dev":
        fallback_path = "config/config.yaml"
        if os.path.exists(fallback_path):
            cfg_path = fallback_path

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        settings = AppSettings(**data)
    else:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # --- Ajustes dinâmicos ---
    # Se provider == ollama, pode sobrescrever base_url via OLLAMA_HOST
    if settings.llm.provider == "ollama":
        ollama_env = os.getenv("OLLAMA_HOST")
        if ollama_env:
            settings.embedding.base_url = ollama_env
            settings.llm.base_url = ollama_env
            settings.llm.api_key = os.getenv("OLLAMA_API_KEY")

    # Se provider == groq, injeta API key do ambiente
    if settings.llm.provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            settings.llm.api_key = groq_key

    if settings.llm.provider == "openrouter":
        or_key = os.getenv("OPENROUTER_API_KEY")
        or_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if or_key:
            settings.llm.api_key = or_key
        settings.llm.base_url = or_base

    return settings
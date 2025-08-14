import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    backend_host: str = "localhost"
    backend_port: int = 8000
    
    ollama_model: str = "gemma3:12b"
    
    litellm_verbose: bool = False
    litellm_drop_params: bool = True
    litellm_suppress_debug_info: bool = True
    
    chroma_persist_directory: str = "./data/vector_db"
    chroma_telemetry_disabled: bool = True
    
    max_file_size_mb: int = 100
    allowed_extensions: list = ["pdf", "docx", "txt", "md"]
    
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    log_level: str = "INFO"
    
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    documents_dir: Path = data_dir / "documents"
    chunks_dir: Path = data_dir / "chunks"
    vector_db_dir: Path = data_dir / "vector_db"
    logs_dir: Path = project_root / "logs"
    
    class Config:
        env_file = ".env"

settings = Settings()

for directory in [settings.data_dir, settings.documents_dir, settings.chunks_dir, settings.vector_db_dir, settings.logs_dir]:
    directory.mkdir(parents=True, exist_ok=True)
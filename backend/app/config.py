from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "rag_agent"

    # SerpAPI
    serpapi_key: str = ""

    # Paths
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data"
    upload_dir: Path = data_dir / "uploads"
    chroma_dir: Path = data_dir / "chroma"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    hybrid_top_k: int = 10

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

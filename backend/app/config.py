from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek (OpenAI-compatible)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    embed_model: str = "BAAI/bge-m3"

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

    # 上下文工程：分层 token 预算
    context_total_tokens: int = 32_000
    context_system_ratio: float = 0.15
    context_memory_ratio: float = 0.10
    context_history_ratio: float = 0.20
    context_retrieved_ratio: float = 0.35
    context_scratchpad_ratio: float = 0.20
    context_summary_max_chars: int = 2_000

    # 启动时预载检索链路（embeddings + Chroma + BM25）。
    # 测试里若用 `with TestClient(app)` 会触发 lifespan，设 PRELOAD_MODELS=false 跳过。
    preload_models: bool = True

    # History compression
    history_window_size: int = 10

    @property
    def embed_model_path(self) -> str:
        local_path = self.data_dir / "models" / "BAAI" / "bge-m3"
        if local_path.exists():
            return str(local_path)
        return self.embed_model

    @property
    def reranker_model_path(self) -> str:
        local_path = self.data_dir / "models" / "BAAI" / "bge-reranker-v2-m3"
        if local_path.exists():
            return str(local_path)
        return self.reranker_model

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

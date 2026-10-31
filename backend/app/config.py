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

    # 上传上限（MB）。上传是流式落盘的，超限时在写入过程中就停手并删掉半成品。
    max_upload_mb: int = 20

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    hybrid_top_k: int = 10

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # 精排开关。CPU 上精排 10 个候选约 1~3 秒，因此它是可关闭的：
    # 关掉即回到纯 RRF 融合顺序。
    enable_reranker: bool = True
    # 送进精排的候选数。精排只在这 N 个里重排，所以它决定精排的耗时与收益上限。
    rerank_candidates: int = 10

    # 跨语言 query 改写：给 BM25 一路追加从语料自动挖掘的英文术语。
    # 语料是英文课件而 query 是中文时，词面匹配完全对不上；
    # 改写后 BM25 才能命中（实测见 docs/tech-notes/verified-facts.md §9）。
    # **只作用于 BM25 一路** —— 向量侧用原 query，多语言对齐已经够用，
    # 追加英文反而会稀释中文语义信号。
    enable_query_rewrite: bool = True

    # Agent 规划节点：先用一次轻量 LLM 调用判断「要不要检索」，
    # 并把口语问题改写成检索友好的 query，再决定是否预检索。
    #
    # **代价是每次问答多一次 LLM 往返**（实测约 1 秒）。
    # 换来的收益是检索 query 更准 —— 旧实现把用户原话直接丢给检索，
    # 「这个算法咋跑的」这种问法召不回「Worklist Algorithm」。
    #
    # 关掉后：不做规划、不预检索，由模型在推理中自主决定是否调工具
    # （即迁移前的行为）。
    enable_planning: bool = True

    # Agent 迭代上限：模型连续调工具不收敛时强制收尾。
    agent_max_iterations: int = 8

    # --- 三层记忆 ---
    # 第二层「长期记忆」：跨会话记住用户偏好与已确认事实。
    enable_long_term_memory: bool = True
    # 记忆召回的最低相似度。
    # **实测校准过，不是拍的**：同一句话换种问法对原句的相似度在
    # 0.88~0.99，而无关问题在 0.24~0.61。「回答要短一点」对
    # 「我偏好简短回答」只有 0.669，所以常见的 0.95 阈值在这里几乎召不回。
    memory_recall_min_score: float = 0.55

    # 第三层「语义缓存」：相似问题直接复用答案，跳过检索与生成。
    enable_semantic_cache: bool = True
    # **0.90 是权衡后的选择，不是抄的**（实测见 verified-facts §11.3）：
    #   0.85 → 正样本全命中，但会把「活跃变量分析怎么定义」和
    #          「活跃变量分析能用来做什么」（相似度 0.899）判成同一个问题
    #   0.90 → 漏掉 1/6 同义改写，但零错命中
    # 缓存错命中 = 立刻给出看着权威的错误答案，代价远高于漏命中（多等十几秒）。
    semantic_cache_threshold: float = 0.90
    # 缓存有效期（小时）。答案会过时：模型升级、文档更新后旧答案就不对了。
    # 0 表示永不过期（不推荐）。
    semantic_cache_ttl_hours: int = 168

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

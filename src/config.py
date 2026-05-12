from pathlib import Path

# 项目根目录与数据目录配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# 环境变量名称
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
SERPAPI_API_KEY_ENV = "SERPAPI_API_KEY"

# 模型配置
# 默认使用 Gemini 2.5 Flash，避免旧模型不可用导致 404
LLM_MODEL = "gemini-2.5-flash"
# Gemini Embedding 1
EMBED_MODEL = "gemini-embedding-001"

# 文本切分与检索参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

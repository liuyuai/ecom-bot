"""电商客服机器人 - 配置管理（多环境）"""
import os


def _detect_env() -> str:
    """检测运行环境：优先环境变量 APP_ENV，默认 dev"""
    return os.environ.get("APP_ENV", "dev").lower()


def _load_dotenv(env: str) -> dict:
    """
    加载 .env 文件，优先级：
    1. 环境变量（最高）
    2. .env.{env}（如 .env.dev / .env.test / .env.prod）
    3. .env（基础默认值）
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_data = {}

    # 先加载基础 .env（默认值）
    base_env = os.path.join(project_root, ".env")
    if os.path.exists(base_env):
        env_data.update(_parse_env_file(base_env))

    # 再加载环境专属 .env.{env}（覆盖默认值）
    env_specific = os.path.join(project_root, f".env.{env}")
    if os.path.exists(env_specific):
        env_data.update(_parse_env_file(env_specific))

    return env_data


def _parse_env_file(path: str) -> dict:
    """解析 .env 文件"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip().strip('"').strip("'")
    return data


# ===== 运行环境（最先确定，后续加载依赖它）=====
APP_ENV = _detect_env()
_DOTENV = _load_dotenv(APP_ENV)


def _get(key, default=""):
    """取值优先级：环境变量 > .env.{env} > .env > default"""
    return os.environ.get(key, _DOTENV.get(key, default))


# ===== 大模型 =====
LLM_API_KEY = _get("LLM_API_KEY")
LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = _get("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.3"))

# ===== Embedding =====
EMBED_API_KEY = _get("EMBED_API_KEY")
EMBED_BASE_URL = _get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL = _get("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")

# ===== 小模型（意图分类用，便宜快速）=====
# 复用硅基流动的 API Key，用便宜的 7B 模型做分类
SMALL_LLM_API_KEY = EMBED_API_KEY
SMALL_LLM_BASE_URL = EMBED_BASE_URL
SMALL_LLM_MODEL = _get("SMALL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SMALL_LLM_CONFIDENCE_THRESHOLD = float(_get("SMALL_LLM_CONFIDENCE_THRESHOLD", "0.7"))

# ===== Reranker =====
RERANK_MODEL = _get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ===== LangSmith =====
LANGCHAIN_TRACING_V2 = _get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = _get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = _get("LANGCHAIN_PROJECT", "ecom-customer-bot")

if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ===== 项目路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 向量库 =====
CHROMA_PATH = os.path.join(PROJECT_ROOT, _get("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = _get("COLLECTION_NAME", "ecom_knowledge")

# ===== 文档 =====
DOCS_DIR = os.path.join(PROJECT_ROOT, _get("DOCS_DIR", "docs"))

# ===== 检索 =====
TOP_K = int(_get("TOP_K", "3"))

# ===== 会话存储 =====
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, _get("SQLITE_DB_PATH", "checkpoints.db"))

# ===== 服务配置 =====
HOST = _get("HOST", "127.0.0.1")
PORT = int(_get("PORT", "8000"))
DEBUG = _get("DEBUG", "false").lower() == "true"
LOG_LEVEL = _get("LOG_LEVEL", "info")

# ===== 请求超时（秒）=====
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "30"))
EMBED_TIMEOUT = float(_get("EMBED_TIMEOUT", "15"))
RERANK_TIMEOUT = float(_get("RERANK_TIMEOUT", "10"))

# ===== 限流 =====
RATE_LIMIT_PER_MINUTE = int(_get("RATE_LIMIT_PER_MINUTE", "30"))

# ===== Redis（生产环境会话存储，不配置则用 SQLite）=====
REDIS_URL = _get("REDIS_URL", "")  # 如 redis://localhost:6379/0

# ===== 输入限制 =====
MAX_MESSAGE_LENGTH = int(_get("MAX_MESSAGE_LENGTH", "2000"))

# ===== 启动检查 =====
if not LLM_API_KEY:
    print(f"[警告][{APP_ENV}] 未设置 LLM_API_KEY")
if not EMBED_API_KEY:
    print(f"[警告][{APP_ENV}] 未设置 EMBED_API_KEY")

print(f"[配置] 当前环境: {APP_ENV} | 模型: {LLM_MODEL} | 端口: {PORT}")

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    project_name: str = os.getenv("PROJECT_NAME", "smartenv_support_system")
    env: str = os.getenv("ENV", "dev")

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:10222/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen2___5-7B-Instruct")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

    milvus_url: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    milvus_timeout: int = int(os.getenv("MILVUS_TIMEOUT", "10"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_DOCS", "pka_docs_v1")
    milvus_vector_dim: int = int(os.getenv("MILVUS_VECTOR_DIM", "1024"))
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")

    es_url: str = os.getenv("ES_URL", "http://127.0.0.1:9200")
    es_index_name: str = os.getenv("ES_INDEX_NAME", "smartenv_chunks_v1")
    es_timeout: int = int(os.getenv("ES_TIMEOUT", "10"))
    
    embedding_model_path: str = os.getenv("EMBEDDING_MODEL_PATH", "")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    reranker_model_path: str = os.getenv("RERANKER_MODEL_PATH", "")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")

    data_dir: str = os.getenv("DATA_DIR", "./data")
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    log_dir: str = os.getenv("LOG_DIR", "./logs")

    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "smartenv-se100-dev")

    enable_startup_warmup: str = os.getenv("ENABLE_STARTUP_WARMUP", "false")

    # 技术支持主链路的并发隔离配置
    support_max_concurrent_requests: int = int(os.getenv("SUPPORT_MAX_CONCURRENT_REQUESTS", "4"))
    support_degrade_concurrent_requests: int = int(os.getenv("SUPPORT_DEGRADE_CONCURRENT_REQUESTS", "3"))
    support_worker_threads: int = int(
        os.getenv("SUPPORT_WORKER_THREADS", os.getenv("SUPPORT_MAX_CONCURRENT_REQUESTS", "4"))
    )
    support_request_timeout: int = int(os.getenv("SUPPORT_REQUEST_TIMEOUT", "150"))
    
    # SQLite 工单库并发写入配置
    sqlite_busy_timeout_ms: int = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000"))
    sqlite_write_max_retries: int = int(os.getenv("SQLITE_WRITE_MAX_RETRIES", "3"))
    sqlite_write_retry_base_seconds: float = float(os.getenv("SQLITE_WRITE_RETRY_BASE_SECONDS", "0.05"))

settings = Settings()
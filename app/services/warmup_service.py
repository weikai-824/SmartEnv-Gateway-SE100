import time
from typing import Any

from app.config.settings import settings
from app.services.embedding_service import embed_text
from app.services.rerank_service import get_reranker_model
from app.db.milvus_client import search_chunk_embeddings


def is_true(value: str | bool | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def should_enable_startup_warmup() -> bool:
    return is_true(settings.enable_startup_warmup)


def record_warmup_item(
    report: dict[str, Any],
    name: str,
    start: float,
    ok: bool,
    error: str | None = None,
) -> None:
    item = {
        "ok": ok,
        "seconds": round(time.perf_counter() - start, 3),
    }
    if error:
        item["error"] = error
    report["items"][name] = item


def warmup_support_runtime() -> dict[str, Any]:
    """
    启动预热：
    1. 加载 embedding 模型，并执行一次 query embedding
    2. 执行一次 Milvus search，预热向量检索链路
    3. 加载 reranker 模型，并执行一次 compute_score

    注意：
    - 这里只预热底层模型和检索链路
    - 不调用 SupportSupervisor
    - 不调用 Agent
    - 不创建工单
    """
    total_start = time.perf_counter()
    report: dict[str, Any] = {
        "enabled": True,
        "items": {},
    }

    warmup_query = "SE-100 云端离线"
    warmup_context = "检查 Wi-Fi 网络、MQTT 服务器地址、CLOUD 指示灯和云端绑定状态。"

    query_embedding: list[float] | None = None

    # 1. 预热 embedding
    start = time.perf_counter()
    try:
        query_embedding = embed_text(warmup_query)
        record_warmup_item(report, "embedding", start, ok=True)
    except Exception as exc:
        record_warmup_item(report, "embedding", start, ok=False, error=str(exc))

    # 2. 预热 Milvus search
    if query_embedding:
        start = time.perf_counter()
        try:
            search_chunk_embeddings(
                query_embedding=query_embedding,
                doc_id=None,
                top_k=1,
            )
            record_warmup_item(report, "milvus_search", start, ok=True)
        except Exception as exc:
            record_warmup_item(report, "milvus_search", start, ok=False, error=str(exc))

    # 3. 预热 reranker
    start = time.perf_counter()
    try:
        reranker_model = get_reranker_model()
        reranker_model.compute_score([[warmup_query, warmup_context]])
        record_warmup_item(report, "reranker", start, ok=True)
    except Exception as exc:
        record_warmup_item(report, "reranker", start, ok=False, error=str(exc))

    report["total_seconds"] = round(time.perf_counter() - total_start, 3)

    print("[StartupWarmup]", report)
    return report
from typing import Any

from langsmith import traceable

from app.db.elasticsearch_client import search_es_chunks


@traceable(
    name="SparseRetrieve",
    run_type="retriever",
    tags=["rag", "sparse", "elasticsearch", "bm25"]
)
def sparse_retrieve_by_query(
        query: str,
        doc_id: str | None = None,
        top_k: int = 5,
) -> dict[str, Any]:
    query = query.strip()

    if not query:
        raise ValueError("query不能为空")

    if top_k <= 0:
        raise ValueError("top_k必须大于0")

    return search_es_chunks(
        query=query,
        doc_id=doc_id,
        top_k=top_k,
    )
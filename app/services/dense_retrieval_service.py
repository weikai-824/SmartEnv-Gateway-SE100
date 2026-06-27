from typing import Any
from app.db.milvus_client import search_chunk_embeddings
from app.services.embedding_service import embed_text
from langsmith import traceable

#根据问题来稠密检索
@traceable(
    name="DenseRetrieve",
    run_type="retriever",
    tags=["rag", "dense", "milvus"]
)
def dense_retrieve_by_query(query:str,doc_id:str|None=None,top_k:int=5) ->dict[str,Any]:
    query=query.strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k<=0:
        raise ValueError('top_k必须大于0')
    query_embedding=embed_text(query)
    search_result=search_chunk_embeddings(
        query_embedding=query_embedding,
        doc_id=doc_id,
        top_k=top_k
    )
    return {
        'total_hits':search_result['total_hits'],
        'hits':search_result['hits']
    }

















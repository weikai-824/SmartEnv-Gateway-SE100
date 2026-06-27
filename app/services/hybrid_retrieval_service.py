from typing import Any
from app.services.sparse_retrieval_service import sparse_retrieve_by_query
from app.services.dense_retrieval_service import dense_retrieve_by_query
from langsmith import traceable

#混合检索
@traceable(
    name="HybridRetrieve",
    run_type="retriever",
    tags=["rag", "hybrid", "rrf"]
)
def hybrid_retrieve_by_query(query:str,doc_id:str|None=None,
                             top_k:int=5,candidate_k:int=20,rrf_k:int=60)->dict[str,Any]:
    query=query.strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k <=0:
        raise ValueError('top_k必须大于零')
    if candidate_k < top_k:
        raise ValueError('candidate_k必须大于等于top_k')

    #分别调用已有sparse/dense service找到
    dense_result=dense_retrieve_by_query(
        query=query,
        doc_id=doc_id,
        top_k=candidate_k
    )
    sparse_result=sparse_retrieve_by_query(
        query=query,
        doc_id=doc_id,
        top_k=candidate_k
    )
    dense_hits=dense_result.get('hits',[])
    sparse_hits=sparse_result.get('hits',[])
    fusion_pool: dict[str, dict[str, Any]] = {}
    #对dense结果进行处理
    for rank,hit in enumerate(dense_hits,start=1):
        chunk_id=hit['chunk_id']
        if not chunk_id:
            continue
        fusion_pool[chunk_id]={
            'hit':hit,
            'rrf_score':1.0/(rrf_k + rank),
            'dense_rank':rank,
            "sparse_rank":None,
            'matched_retrievers':['dense']
        }
    #对sparse结果进行处理
    for rank,hit in enumerate(sparse_hits,start=1):
        chunk_id=hit.get('chunk_id')
        if not chunk_id:
            continue
        #如果这个chunk_id已经被召回过就在此基础上相加
        rrf_score=1.0/(rank + rrf_k)
        if chunk_id in fusion_pool:
            fusion_pool[chunk_id]['rrf_score']+=rrf_score
            fusion_pool[chunk_id]['sparse_rank']=rank
            fusion_pool[chunk_id]['matched_retrievers'].append('sparse')
        else:
            fusion_pool[chunk_id] = {
                'hit': hit,
                'rrf_score': 1 / (rrf_k + rank),
                'dense_rank': None,
                "sparse_rank": rank,
                'matched_retrievers': ['sparse']
            }
    fusion_result=sorted(fusion_pool.values(),key=lambda item:item['rrf_score'],reverse=True)
    #格式化返回，对齐稀疏和稠密的返回结果
    format_hits=[]
    for rank,item in enumerate(fusion_result[:top_k],start=1):
        hit=item['hit']
        one_hit={
                "rank": rank,
                "rrf_score": float(item["rrf_score"]),
                "dense_rank": item["dense_rank"],
                "sparse_rank": item["sparse_rank"],
                "matched_retrievers": item["matched_retrievers"],
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
                "chunk_index": hit.get("chunk_index"),
                "text": hit.get("text"),
                "char_start": hit.get("char_start"),
                "char_end": hit.get("char_end"),
                "file_name": hit.get("file_name"),
                "source": hit.get("source"),
            }
        format_hits.append(one_hit)

    return {
            "dense_total_hits": dense_result.get("total_hits", 0),
            "sparse_total_hits": sparse_result.get("total_hits", 0),
            "total_hits": len(format_hits),
            "hits": format_hits,
        }









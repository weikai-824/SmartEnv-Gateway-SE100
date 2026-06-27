from pathlib import Path
from typing import Any
from app.config.settings import Settings
from app.services.hybrid_retrieval_service import hybrid_retrieve_by_query
from FlagEmbedding import FlagReranker
from langsmith import traceable

reranker_model_path=Settings.reranker_model_path
#全局模型对象，避免每次请求都要重新加载
_reranker_model=None

#加载reranker模型
def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        model_path = Settings.reranker_model_path
        if not model_path:
            raise ValueError("RERANKER_MODEL_PATH 未配置，请检查项目根目录下的 .env 文件")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Reranker 模型路径不存在: {model_path}")

        _reranker_model = FlagReranker(
            model_path,
            use_fp16=str(Settings.reranker_device).startswith("cuda"),
        )
    return _reranker_model

# 构造 rerank 检索结果
def build_rerank_result(
    hits: list[dict[str, Any]],
    hybrid_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        'hybrid_total_hits': hybrid_result.get('total_hits', 0),
        'dense_total_hits': hybrid_result.get('dense_total_hits', 0),
        'sparse_total_hits': hybrid_result.get('sparse_total_hits', 0),
        'total_hits': len(hits),
        'hits': hits,
    }

#rerank重排混合检索后的召回chunk
@traceable(
    name="HybridRerankRetrieve",
    run_type="retriever",
    tags=["rag", "hybrid", "rerank"]
)
def hybrid_rerank_retrieve_by_query(
        query:str,
        doc_id:str|None=None,
        top_k:int=5,
        candidate_k:int=20,
        rrf_k:int=60
)->dict[str,Any]:
    query=query.strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k<=0:
        raise ValueError('top_k必须大于零')
    if candidate_k<top_k:
        raise ValueError('candidate_k必须大于等于top_k')
    #1.先用hybrid召回更大的候选池
    hybrid_result=hybrid_retrieve_by_query(
        doc_id=doc_id,
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k
    )
    candidate_hits=hybrid_result.get('hits')
    if not candidate_hits:
        return build_rerank_result(
            hits=[],
            hybrid_result=hybrid_result,
        )
    #2.组装reranker输入pair['query','text']
    pairs=[]
    valid_hits=[]
    for hit in candidate_hits:
        text = str(hit.get('text') or '').strip()
        if not text:
            continue
        pairs.append([query,text])
        valid_hits.append(hit)
    if not pairs:
        return build_rerank_result(
            hits=[],
            hybrid_result=hybrid_result,
        )

    #3.reranker打分
    reranker_model=get_reranker_model()
    rerank_scores=reranker_model.compute_score(pairs)
    if hasattr(rerank_scores,'tolist'):
        rerank_scores=rerank_scores.tolist()
    if isinstance(rerank_scores,(int,float)):
        rerank_scores=[rerank_scores]

    #4.根据reranker的分数重新构造hit结构
    reranked_hits=[]
    for hit, score in zip(valid_hits, rerank_scores):
        new_hit = dict(hit)
        new_hit['hybrid_rank'] = hit['rank']
        new_hit['rerank_score'] = float(score)
        reranked_hits.append(new_hit)
    reranked_hits.sort(key=lambda item: item['rerank_score'], reverse=True)

    #5.根据top_k截断，并刷新最新rank
    final_hits=[]
    for rank,hit in enumerate(reranked_hits[:top_k],start=1):
        hit['rank']=rank
        hit['rerank_rank']=rank
        final_hits.append(hit)
    return build_rerank_result(
        hits=final_hits,
        hybrid_result=hybrid_result,
    )












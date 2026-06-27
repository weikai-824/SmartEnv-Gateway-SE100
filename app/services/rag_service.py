from typing import Any
from app.services.rerank_service import hybrid_rerank_retrieve_by_query
from app.services.context_builder_service import build_context_from_hits
from app.services.llm_service import generate_answer
from app.services.query_planner_service import plan_query,get_retrieval_query
import re
from langsmith import traceable

citation_re = re.compile(r'\[(C\d+)\]')
bracket_citation_re = re.compile(r'\[([A-Za-z]+\d+)\]')
#1.校验并清理答案中的引用编号。
def check_citation_id(answer:str,contexts:list[dict[str,Any]]) ->dict[str,Any]:
    answer=answer or ''
    valid_ids=[context.get('citation_id') for context in contexts if context.get('citation_id')]
    used_ids=list(dict.fromkeys(citation_re.findall(answer)))
    invalid_ids=[citation_id for citation_id in used_ids if citation_id not in valid_ids]

    cleaned_answer=answer
    unsupported_citation_tokens: list[str] = []
    # 清理 [Q15]、[FAQ3] 这类非 C 类伪引用
    for token in bracket_citation_re.findall(answer):
        if re.fullmatch(r"C\d+", token):
            continue
        bracket_token = f"[{token}]"
        if bracket_token not in unsupported_citation_tokens:
            unsupported_citation_tokens.append(bracket_token)

        cleaned_answer = cleaned_answer.replace(bracket_token, "")

    # 清理不存在于当前 contexts 的 [C数字] 引用
    for citation_id in invalid_ids:
        cleaned_answer=cleaned_answer.replace(f'[{citation_id}]','')
    cleaned_answer=cleaned_answer.strip()
    # 清理后重新统计最终答案里的合法引用
    used_ids = list(dict.fromkeys(citation_re.findall(cleaned_answer)))

    # 有上下文但答案没引用时，补一个最小引用
    citation_was_added = False
    if valid_ids and cleaned_answer and not used_ids:
        cleaned_answer = f"{cleaned_answer} [{valid_ids[0]}]"
        used_ids = [valid_ids[0]]
        citation_was_added = True

    return {
        "answer": cleaned_answer,
        "citation_check": {
            "valid_citation_ids": valid_ids,
            "used_citation_ids": used_ids,
            "invalid_citation_ids": invalid_ids,
            "unsupported_citation_tokens": unsupported_citation_tokens,
            "citation_was_added": citation_was_added,
            "is_valid": len(invalid_ids) == 0 and len(unsupported_citation_tokens) == 0,
            "answer_was_cleaned": cleaned_answer != answer,
        }
    }

#2.构建检索摘要
def build_retrieval_summary(
        retrieval_result: dict[str, Any],
        raw_hits: int = 0,
        filtered_hits: int = 0
) -> dict[str, Any]:
    return {
        "raw_hits": raw_hits,
        "filtered_hits": filtered_hits,
        "total_hits": retrieval_result.get("total_hits", 0),
        "hybrid_total_hits": retrieval_result.get("hybrid_total_hits", 0),
        "dense_total_hits": retrieval_result.get("dense_total_hits", 0),
        "sparse_total_hits": retrieval_result.get("sparse_total_hits", 0),
    }

#3.构造无上下文时的标准RAG返回，把RAG查询失败从'黑盒失败'变成'可诊断失败'
def build_no_context_response(
        query: str,
        retrieval_result: dict[str, Any],
        no_context_reason: str,
        raw_hits: int = 0,
        filtered_hits: int = 0,
        query_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
    "query": query,
    "query_plan": query_plan,
    "answer": "当前知识库没有检索到足够相关内容，无法基于资料回答。",
    "has_context": False,
    "retrieval_status": "no_context",
    "no_context_reason": no_context_reason,
    "contexts": [],
    "retrieval": build_retrieval_summary(
        retrieval_result=retrieval_result,
        raw_hits=raw_hits,
        filtered_hits=filtered_hits
    ),
    "llm": None,
}

#4.根据rerank_score过滤低相关hits
def filter_hits_by_rerank_score(
        hits:list[dict[str,Any]],
        min_rerank_score:float|None=None
) ->list[dict[str,Any]]:
    if min_rerank_score==None:
        return hits
    filtered_hits=[]
    for hit in hits:
        rerank_score=hit.get('rerank_score')
        if rerank_score==None:
            continue
        try:
            rerank_score=float(rerank_score)
        except(TypeError,ValueError):
            continue
        if rerank_score>=min_rerank_score:
            filtered_hits.append(hit)
    return filtered_hits

#5.RAG主链路
@traceable(
    name="AnswerByRAG",
    run_type="chain",
    tags=["rag", "answer"]
)
def answer_by_rag(
        query:str,
        doc_id:str|None=None,
        top_k:int=5,
        candidate_k:int=20,
        rrf_k:int=60,
        max_context_chars:int=6000,
        max_chunk_chars:int=1200,
        min_rerank_score: float | None = None
)->dict[str,Any]:
    query=str(query or '').strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k<=0:
        raise ValueError('top_k必须大于零')
    if candidate_k<top_k:
        raise ValueError('candidate_k必须大于等于top_k')

    query_plan=plan_query(query)
    retrieval_query=get_retrieval_query(query_plan,fallback_query=query)

    #1.检索hybrid+rerank得到hits
    retrieval_result=hybrid_rerank_retrieve_by_query(
        query=retrieval_query,
        doc_id=doc_id,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k
    )
    raw_hits=retrieval_result.get('hits',[])
    #状况1：检索完全没命中，不调用llm
    if not raw_hits:
        return  build_no_context_response(
        query=query,
        retrieval_result=retrieval_result,
        no_context_reason="no_retrieval_hits",
        raw_hits=0,
        filtered_hits=0,
        query_plan=query_plan
    )
    #状况2：所有检索命中的hits相关性都太低全被过滤掉了
    hits=filter_hits_by_rerank_score(hits=raw_hits,min_rerank_score=min_rerank_score)
    if not hits:
        return build_no_context_response(
            query=query,
            retrieval_result=retrieval_result,
            no_context_reason="all_hits_below_min_rerank_score",
            raw_hits=len(raw_hits),
            filtered_hits=0,
            query_plan=query_plan
        )
    #2.根据hits构造上下文
    context_result=build_context_from_hits(
        hits=hits,
        max_chunk_chars=max_chunk_chars,
        max_context_chars=max_context_chars
    )
    context_text=context_result.get('context_text','').strip()
    contexts=context_result.get('contexts',[])
    #状况3：有hits，但是构造不出上下文context，也不调用llm
    if not context_text:
        return build_no_context_response(
            query=query,
            retrieval_result=retrieval_result,
            no_context_reason="empty_context_after_context_builder",
            raw_hits=len(raw_hits),
            filtered_hits=len(hits),
            query_plan=query_plan
        )
    #3.根据上下文和问题调用RAG链生成答案
    llm_result=generate_answer(
        query=query,
        context_text=context_text
    )
    citation_result=check_citation_id(answer=llm_result.get('answer',''),contexts=contexts)
    return {
        "query": query,
        "query_plan": query_plan,
        "answer": citation_result.get("answer", ""),
        "has_context": bool(contexts),
        "contexts": contexts,
        "citation_check": citation_result["citation_check"],
        "retrieval_status": "success",
        "no_context_reason": None,
        "retrieval": build_retrieval_summary(
            retrieval_result=retrieval_result,
            raw_hits=len(raw_hits),
            filtered_hits=len(hits)
        ),
        "llm": {
            "model": llm_result.get("model"),
        }
    }

#6.只做检索与证据构造，不调用 generate_answer
@traceable(
    name="RetrieveEvidenceByRAG",
    run_type="retriever",
    tags=["rag", "retrieval", "hybrid-rerank"]
)
def retrieve_evidence_by_rag(
        query: str,
        doc_id: str | None = None,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
        max_context_chars: int = 6000,
        max_chunk_chars: int = 1200,
        min_rerank_score: float | None = None
) -> dict[str, Any]:
    query = str(query or '').strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k <= 0:
        raise ValueError('top_k必须大于零')
    if candidate_k < top_k:
        raise ValueError('candidate_k必须大于等于top_k')

    query_plan = plan_query(query)
    retrieval_query = get_retrieval_query(query_plan, fallback_query=query)
    retrieval_result = hybrid_rerank_retrieve_by_query(
        query=retrieval_query,
        doc_id=doc_id,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k
    )

    raw_hits = retrieval_result.get('hits', [])
    if not raw_hits:
        return build_no_context_response(
            query=query,
            retrieval_result=retrieval_result,
            no_context_reason="no_retrieval_hits",
            raw_hits=0,
            filtered_hits=0,
            query_plan=query_plan
        )

    hits = filter_hits_by_rerank_score(
        hits=raw_hits,
        min_rerank_score=min_rerank_score
    )
    if not hits:
        return build_no_context_response(
            query=query,
            retrieval_result=retrieval_result,
            no_context_reason="all_hits_below_min_rerank_score",
            raw_hits=len(raw_hits),
            filtered_hits=0,
            query_plan=query_plan
        )

    context_result = build_context_from_hits(
        hits=hits,
        max_chunk_chars=max_chunk_chars,
        max_context_chars=max_context_chars
    )
    contexts = context_result.get('contexts', [])
    context_text = context_result.get('context_text', '').strip()
    if not context_text:
        return build_no_context_response(
            query=query,
            retrieval_result=retrieval_result,
            no_context_reason="empty_context_after_context_builder",
            raw_hits=len(raw_hits),
            filtered_hits=len(hits),
            query_plan=query_plan
        )

    return {
        "query": query,
        "query_plan": query_plan,
        "answer": "",
        "has_context": bool(contexts),
        "contexts": contexts,
        "citation_check": None,
        "retrieval_status": "success",
        "no_context_reason": None,
        "retrieval": build_retrieval_summary(
            retrieval_result=retrieval_result,
            raw_hits=len(raw_hits),
            filtered_hits=len(hits)
        ),
        "llm": None,
    }










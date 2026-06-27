from app.services.json_output_service import extract_json_object
from functools import lru_cache
from typing import Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from app.config.settings import Settings
from app.services.rag_diag_tool import build_rag_diagnose_tool
from app.services.rag_service import check_citation_id
from app.schemas.rag_diag import RagDiagAgentFinal
from langsmith import traceable

#1.创建系统提示词
rag_agent_system_prompt = """
你是 SmartEnv-Gateway SE-100 技术支持诊断 Agent。

你的任务：
基于 rag_diagnose 工具返回的知识库证据，生成面向用户的故障排查回答和结构化诊断结果。

规则：
1. 用户问题只要与 SE-100 产品、配置、联网、指示灯、云端平台、OTA、传感器或故障排查有关，必须调用 rag_diagnose 工具。
2. 不要使用模型自身知识回答 SE-100 的事实问题。
3. rag_diagnose 工具只返回知识库证据，不负责生成最终回答。
4. 你必须基于工具返回的 contexts 生成 final_answer 和 diagnosis。
5. 不要编造引用编号，只能使用工具返回 contexts 中已有的 citation_id，例如 C1、C2。
6. final_answer 中的关键结论后要保留必要引用，例如 [C1]、[C2]。
7. 如果工具返回 has_context=false，说明知识库证据不足，不要强行诊断。
8. 最终只输出一个 JSON 对象，不要输出 Markdown，不要输出代码块。
9. final_answer 要简洁，不要重复大段证据原文。

最终 JSON 格式：
{
  "final_answer": "给用户看的简洁排障回答，保留必要引用，例如 [C1] [C2]",
  "diagnosis": {
    "problem_type": "云端连接问题",
    "known_symptoms": ["用户已经明确描述的现象"],
    "possible_causes": ["基于工具证据得到的可能原因"],
    "priority_steps": ["优先排查步骤"],
    "missing_information": ["还需要用户补充的信息"],
    "need_after_sales": false,
    "evidence_citations": ["C1", "C2"]
  }
}

problem_type 只能从下面选择一个：
供电问题、联网问题、云端连接问题、平台离线/绑定问题、数据不上报问题、本地配置页问题、OTA升级问题、告警问题、外接传感器问题、恢复出厂设置问题、产品边界问题、信息不足。
"""

#2.创造大语言模型
def build_agent_model() ->ChatOpenAI:
    chat_llm=ChatOpenAI(
        model=Settings.llm_model,
        base_url=Settings.llm_base_url,
        api_key=Settings.llm_api_key,
        temperature=0,
        timeout=Settings.llm_timeout,
        max_tokens=500
    )
    return chat_llm

#3.从 messages 里找 RAG 工具返回结果
def find_rag_result(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        rag_result = extract_json_object(getattr(message, 'content', None))
        if not rag_result:
            continue
        if "retrieval_status" in rag_result and "contexts" in rag_result:
            return rag_result
    return None

#4.Agent 工具调用记录
def collect_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    ignored_tool_names = {"RagDiagAgentFinal"}
    tool_calls = []
    for message in messages:
        for tool_call in getattr(message, 'tool_calls', []) or []:
            name=tool_call.get('name')
            if not name or name in ignored_tool_names:
                continue
            tool_calls.append({'name':name})
    return tool_calls


#5.把 LangChain structured_response 转成普通 dict
def structured_response_to_dict(data: Any) -> dict[str, Any] | None:
    if data is None:
        return None
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, dict):
        return data
    return None

#6.清理 diagnosis.evidence_citations 里不存在的引用编号
def clean_diagnosis_citations(
        diagnosis: dict[str, Any] | None,
        contexts: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(diagnosis, dict):
        return diagnosis

    valid_ids = {
        context.get("citation_id")
        for context in contexts
        if context.get("citation_id")
    }
    raw_citations = diagnosis.get("evidence_citations") or []
    diagnosis["evidence_citations"] = [
        citation_id
        for citation_id in raw_citations
        if citation_id in valid_ids
    ]
    return diagnosis

#7.从 Agent 最后一条消息里解析最终 JSON，并用 Pydantic 校验结构
def get_agent_final_output(messages:list[Any]) ->dict[str,Any]|None:
    if not messages:
        return None
    final_message=messages[-1]
    data=extract_json_object(getattr(final_message,'content',None))

    if not data:
        return None

    try:
        return RagDiagAgentFinal.model_validate(data).model_dump()
    except Exception:
        return None

#8.提前缓存agent
@lru_cache(maxsize=16)
def get_rag_diag_agent(
    doc_id: str | None,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    max_context_chars: int,
    max_chunk_chars: int,
    min_rerank_score: float | None,
):
    rag_tool = build_rag_diagnose_tool(
        doc_id=doc_id,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        max_chunk_chars=max_chunk_chars,
        max_context_chars=max_context_chars,
        min_rerank_score=min_rerank_score,
    )

    return create_agent(
        model=build_agent_model(),
        system_prompt=rag_agent_system_prompt,
        tools=[rag_tool],
        response_format=ToolStrategy(RagDiagAgentFinal),
    )

#9.agent运行主逻辑,创建 Agent、执行 Agent、整理返回
@traceable(
    name="RagDiagAgent",
    run_type="chain",
    tags=["agent", "rag"]
)
def run_rag_diag_agent(
        query:str,
        doc_id:str|None=None,
        top_k:int=5,
        candidate_k:int=20,
        rrf_k:int=60,
        max_context_chars: int = 6000,
        max_chunk_chars: int = 1200,
        min_rerank_score: float | None = None
) ->dict[str,Any]:
    query=str(query or '').strip()
    if not query:
        raise ValueError('query不能为空')
    agent = get_rag_diag_agent(
        doc_id,
        top_k,
        candidate_k,
        rrf_k,
        max_context_chars,
        max_chunk_chars,
        min_rerank_score,
    )
    agent_state = agent.invoke(
        {
            'messages': [
                {'role': 'user', 'content': query}
            ]
        },
        config={
            "run_name": "RagDiagAgentInvoke",
            "tags": ["rag-agent", "tool-calling"],
            "metadata": {
                "doc_id": doc_id,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "rrf_k": rrf_k,
                "min_rerank_score": min_rerank_score
            }
        }
    )
    messages = agent_state.get('messages', [])

    rag_result = find_rag_result(messages)
    tool_calls = collect_tool_calls(messages)
    agent_final_output = structured_response_to_dict(
        agent_state.get('structured_response')
    )

    if agent_final_output is None:
        agent_final_output = get_agent_final_output(messages)

    if rag_result is None:
        return {
            "tool_call_count": len(tool_calls),
            "tool_calls": tool_calls,
            "final_answer": "RAG 诊断 Agent 未调用 rag_diagnose 工具，本次回答已被拦截。请重新发起请求或检查 tool calling 配置。",
            "diagnosis": None,
            "has_context": False,
            "contexts": [],
            "citation_check": None,
            "retrieval_status": "agent_no_rag_result",
            "no_context_reason": "agent_did_not_call_rag_diagnose_tool",
        }

    contexts = rag_result.get("contexts", [])
    has_context = rag_result.get("has_context", False)
    final_answer = rag_result.get("answer") or ""
    diagnosis = None

    if agent_final_output:
        final_answer = agent_final_output.get("final_answer") or final_answer
        diagnosis = agent_final_output.get("diagnosis")
    if not final_answer:
        if has_context:
            final_answer = "已检索到相关知识库证据，但 Agent 未生成最终诊断回答。请检查结构化输出配置。"
        else:
            final_answer = "当前知识库没有检索到足够相关内容，无法基于资料回答。"

    citation_check = None
    if has_context and final_answer:
        citation_result = check_citation_id(
            answer=final_answer,
            contexts=contexts
        )
        final_answer = citation_result.get("answer", final_answer)
        citation_check = citation_result.get("citation_check")

    diagnosis = clean_diagnosis_citations(
        diagnosis=diagnosis,
        contexts=contexts
    )
    return {
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "diagnosis": diagnosis,
        "has_context": has_context,
        "contexts": contexts,
        "citation_check": citation_check,
        "retrieval_status": rag_result.get("retrieval_status"),
        "no_context_reason": rag_result.get("no_context_reason"),
    }























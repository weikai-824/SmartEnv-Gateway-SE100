from typing import Any
from langchain.tools import tool
from app.services.rag_service import retrieve_evidence_by_rag

#1.截断文本防止上下文过长
def shorten_text(text: Any, max_chars: int = 650) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

#2.压缩单条证据，避免 Agent 第二轮 LLM 输入过大
def slim_context(context: dict[str, Any], max_text_chars: int = 650) -> dict[str, Any]:
    return {
        "context_id": context.get("context_id"),
        "citation_id": context.get("citation_id"),
        "chunk_id": context.get("chunk_id"),
        "source": context.get("source"),
        "text": shorten_text(context.get("text"), max_text_chars),
    }

#3.只保留 Agent 诊断需要的 query_plan 字段
def slim_query_plan(query_plan: dict[str, Any] | None) -> dict[str, Any] | None:

    if not isinstance(query_plan, dict):
        return None

    return {
        "planning_status": query_plan.get("planning_status"),
        "intent": query_plan.get("intent"),
        "normalized_query": query_plan.get("normalized_query"),
        "retrieval_query": query_plan.get("retrieval_query"),
        "key_terms": query_plan.get("key_terms", []),
        "missing_information": query_plan.get("missing_information", []),
    }

#4.压缩 RAG 证据结果
def slim_evidence_result(rag_result: dict[str, Any]) -> dict[str, Any]:
    contexts = rag_result.get("contexts") or []

    return {
        "query": rag_result.get("query"),
        "query_plan": slim_query_plan(rag_result.get("query_plan")),
        "answer": rag_result.get("answer") or "",
        "has_context": rag_result.get("has_context", False),
        "contexts": [
            slim_context(context, max_text_chars=650)
            for context in contexts[:3]
        ],
        "citation_check": None,
        "retrieval_status": rag_result.get("retrieval_status"),
        "no_context_reason": rag_result.get("no_context_reason"),
        "retrieval": rag_result.get("retrieval"),
    }

#5.工具主业务逻辑
def build_rag_diagnose_tool(
    doc_id: str | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    max_context_chars: int = 6000,
    max_chunk_chars: int = 1200,
    min_rerank_score: float | None = None,
):
    @tool("rag_diagnose")
    def rag_diagnose_tool(query: str) -> dict[str, Any]:
        """
        查询 SmartEnv-Gateway SE-100 产品知识库，返回用于诊断的文档证据。

        当用户问题涉及 SE-100 的产品能力、故障排查、联网配置、指示灯状态、
        云端平台离线、本地 Web 配置页、MQTT 参数、OTA 升级、告警、外接传感器、
        恢复出厂设置或技术支持处理建议时，应调用本工具。

        输入参数：
        - query：用户关于 SE-100 的原始问题或经整理后的故障描述。

        返回结果：
        - has_context：是否检索到可用证据。
        - contexts：支持诊断的知识库证据片段，每条包含 citation_id。
        - query_plan：查询规划信息。
        - retrieval_status：检索状态。
        - no_context_reason：没有可用上下文时的原因。

        注意：
        - 本工具不生成最终回答。
        - 最终回答、故障类型、排查步骤和售后判断应由 RagDiagAgent 基于 contexts 生成。
        - Agent 只能使用 contexts 中已有的 citation_id，例如 C1、C2。
        """
        rag_result = retrieve_evidence_by_rag(
            query=query,
            doc_id=doc_id,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            max_chunk_chars=max_chunk_chars,
            max_context_chars=max_context_chars,
            min_rerank_score=min_rerank_score,
        )

        return slim_evidence_result(rag_result)

    return rag_diagnose_tool
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatContext, CitationCheck, ChatQueryRequest


class RagDiagAgentRequest(ChatQueryRequest):
    """RAG 诊断 Agent 请求体"""

    pass


class AgentToolCall(BaseModel):
    """Agent 工具调用记录"""

    name: str | None = None


ProblemType = Literal[
    "供电问题",
    "联网问题",
    "云端连接问题",
    "平台离线/绑定问题",
    "数据不上报问题",
    "本地配置页问题",
    "OTA升级问题",
    "告警问题",
    "外接传感器问题",
    "恢复出厂设置问题",
    "产品边界问题",
    "信息不足",
]


class RagDiagnosis(BaseModel):
    """RAG 诊断 Agent 的结构化故障诊断结果"""

    problem_type: ProblemType = Field(
        default="信息不足",
        description="故障类型，只能从预设类型中选择。"
    )
    known_symptoms: list[str] = Field(
        default_factory=list,
        description="用户已经明确描述的故障现象。"
    )
    possible_causes: list[str] = Field(
        default_factory=list,
        description="基于 RAG 工具结果得到的可能原因。"
    )
    priority_steps: list[str] = Field(
        default_factory=list,
        description="建议用户优先执行的排查步骤。"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="当前还需要用户补充的信息。"
    )
    need_after_sales: bool = Field(
        default=False,
        description="是否建议转人工售后或维修。"
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="诊断依据引用，只能使用 RAG contexts 中已有的 citation_id，例如 C1、C2。"
    )


class RagDiagAgentFinal(BaseModel):
    """RAG 诊断 Agent 最终结构化输出"""

    final_answer: str = Field(
        description="给用户看的简洁故障排查回答，必须基于 RAG 工具结果，必要时保留 [C1] 形式引用。"
    )
    diagnosis: RagDiagnosis = Field(
        description="结构化故障诊断结果。"
    )


class RagDiagAgentResponse(BaseModel):
    """RAG 诊断 Agent 响应体"""

    tool_call_count: int
    tool_calls: list[AgentToolCall] = Field(default_factory=list)

    final_answer: str
    diagnosis: RagDiagnosis | None = None

    has_context: bool
    contexts: list[ChatContext] = Field(default_factory=list)
    citation_check: CitationCheck | None = None

    retrieval_status: str | None = None
    no_context_reason: str | None = None
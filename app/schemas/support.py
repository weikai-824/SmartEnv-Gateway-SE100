from typing import Any
from pydantic import BaseModel, Field
from app.schemas.chat import ChatQueryRequest


class SupportSupervisorRequest(ChatQueryRequest):
    """统一技术支持入口请求体"""

    session_id: str | None = Field(
        default=None,
        description="会话 ID；第一次请求可以不传，系统会自动生成。",
    )
    ticket_id: str | None = Field(
        default=None,
        description="已有工单 ID；查询、更新、追加备注时可以传。",
    )


class SupportToolCall(BaseModel):
    """Supervisor 汇总后的工具调用记录"""

    agent: str | None = Field(default=None, description="调用工具的 Agent 名称。")
    tool: str | None = Field(default=None, description="工具名称。")


class SupportSupervisorResponse(BaseModel):
    """统一技术支持入口响应体"""

    session_id: str = Field(description="当前会话 ID。")
    intent: str = Field(description="Supervisor 判断出的用户意图。")
    stage: str = Field(description="当前业务阶段。")

    final_answer: str = Field(description="给用户看的最终回复。")
    next_action: str = Field(description="建议用户下一步怎么做。")

    active_ticket_id: str | None = Field(
        default=None,
        description="当前关联的工单 ID；没有工单时为空。",
    )

    tool_call_count: int = Field(description="本次真实业务工具调用次数。")
    tool_calls: list[SupportToolCall] = Field(
        default_factory=list,
        description="本次调用过的真实业务工具。",
    )

    diagnosis: dict[str, Any] | None = Field(
        default=None,
        description="RAG 诊断 Agent 的结构化诊断结果。",
    )
    ticket: dict[str, Any] | None = Field(
        default=None,
        description="TicketAgent 对外展示的工单摘要。",
    )















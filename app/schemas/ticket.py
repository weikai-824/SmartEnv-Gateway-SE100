from typing import Any, Literal
from pydantic import BaseModel, Field


TicketStatus = Literal[
    "待处理",
    "待用户补充",
    "待工程师处理",
    "已解决",
    "已关闭",
    "已升级人工",
]

TicketPriority = Literal[
    "P0",
    "P1",
    "P2",
    "P3",
]

TicketActionType = Literal[
    "创建工单",
    "查询工单",
    "更新状态",
    "追加备注",
    "无动作",
]

class TicketAgentRequest(BaseModel):
    """工单 Agent 请求体"""

    query: str = Field(description="用户关于工单创建、查询、更新或追加备注的请求。")
    ticket_id: str | None = Field(default=None, description="已有工单 ID，没有则不传。")

class TicketToolCall(BaseModel):
    """工单 Agent 工具调用记录"""

    name: str | None = Field(default=None, description="工具名称。")

class TicketAction(BaseModel):
    """工单 Agent 本次执行的动作结果"""

    action_type: TicketActionType = Field(description="本次执行的工单动作。")

    ticket_id: str | None = Field(default=None, description="工单 ID。")
    status: TicketStatus | None = Field(default=None, description="工单状态。")
    priority: TicketPriority | None = Field(default=None, description="工单优先级。")

    summary: str = Field(default="", description="给客服或工程师看的工单摘要。")
    next_required_info: list[str] = Field(
        default_factory=list,
        description="还需要用户补充的信息。",
    )

class TicketAgentFinal(BaseModel):
    """工单 Agent 最终结构化输出"""

    final_answer: str = Field(description="给用户看的简洁回复。")
    action: TicketAction = Field(description="本次工单动作。")

class TicketAgentResponse(BaseModel):
    """工单 Agent 接口响应体"""

    tool_call_count: int = Field(description="本次 Agent 调用工具的次数。")
    tool_calls: list[TicketToolCall] = Field(
        default_factory=list,
        description="本次 Agent 调用过的工具列表。",
    )

    final_answer: str = Field(description="给用户看的最终回复。")
    action: TicketAction | None = Field(default=None, description="结构化工单动作。")

    ticket: dict[str, Any] | None = Field(
        default=None,
        description="工具返回的完整工单详情。",
    )
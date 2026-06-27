import re
import json
from typing import Any
from langchain.tools import tool
from app.services.ticket_service import (
    add_ticket_note,
    create_ticket,
    get_ticket_detail,
    update_ticket_status
)

# 把模型传入的缺失信息统一整理成 list[str]
def normalize_missing_information(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    text = str(value or "").strip()
    if not text:
        return []
    # 兼容模型偶尔传入 JSON 数组字符串
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [
            str(item).strip()
            for item in data
            if str(item).strip()
        ]
    # 兜底：只按换行和分号切，不按逗号/顿号切
    parts = re.split(r"[\n；;]+", text)
    return[part.strip() for part in parts if part.strip()]

#构建工单agent可调用的工具列表
def build_ticket_tools() ->list[Any]:

    @tool('create_ticket')
    def create_ticket_tool(
            title: str,
            problem_type: str = "信息不足",
            priority: str = "P2",
            summary: str = "",
            missing_information: list[str] | None = None,
            source_query: str = "",
            status: str = "待处理",
    ) ->dict[str,Any]:
        """
        创建一个技术支持工单。

        适用场景：
        - 用户明确要求创建工单。
        - 用户的问题需要人工售后继续跟进。
        - 用户已经提供了故障现象，需要沉淀为工单。
        - RAG 诊断后仍需要用户补充信息或工程师介入。

        参数说明：
        - title：工单标题，必须简洁。
        - problem_type：问题类型，例如 云端连接问题、联网问题、数据不上报问题、信息不足。
        - priority：优先级，只能是 P0、P1、P2、P3。默认 P2。
        - summary：给客服或工程师看的问题摘要。
        - missing_information：必须传 list[str]。每个元素是一条完整缺失信息，不要把一个完整问题拆碎。
        - source_query：用户原始请求。
        - status：工单状态，只能是 待处理、待用户补充、待工程师处理、已解决、已关闭、已升级人工。
        """
        try:
            ticket=create_ticket(
                title=title,
                problem_type=problem_type,
                priority=priority,
                summary=summary,
                missing_information=normalize_missing_information(missing_information),
                source_query=source_query,
                status=status
            )
            return {
                'ok':True,
                'action_type':'创建工单',
                'ticket':ticket
            }
        except Exception as e:
            return {
                'ok':False,
                'action_type':'创建工单',
                'error':str(e)
            }

    @tool('get_ticket_detail')
    def get_ticket_detail_tool(ticket_id:str) ->dict[str,Any]:
        """
        查询已有工单详情。

        适用场景：
        - 用户询问某个工单的处理状态。
        - 用户提供 ticket_id，希望查看工单记录。
        - Agent 需要确认工单当前状态或备注。
        """
        try:
            ticket=get_ticket_detail(ticket_id=ticket_id)
            return {
                'ok':True,
                'action_type':'查询工单',
                'ticket':ticket
            }
        except Exception as e:
            return {
                'ok': False,
                'action_type': '查询工单',
                'error': str(e)
            }

    @tool('update_ticket_status')
    def update_ticket_status_tool(ticket_id:str,status:str) ->dict[str,Any]:
        """
        更新工单状态。

        适用场景：
        - 用户要求关闭工单。
        - 用户表示问题已经解决。
        - 用户要求升级人工处理。
        - 客服或工程师需要修改工单流转状态。

        status 只能使用：
        - 待处理
        - 待用户补充
        - 待工程师处理
        - 已解决
        - 已关闭
        - 已升级人工
        """
        try:
            ticket=update_ticket_status(
                ticket_id=ticket_id,
                status=status
            )
            return {
                'ok': True,
                'action_type': '更新状态',
                'ticket': ticket
            }
        except Exception as e:
            return {
                'ok': False,
                'action_type': '更新状态',
                'error': str(e)
            }

    @tool("add_ticket_note")
    def add_ticket_note_tool(ticket_id: str, note: str) ->dict[str,Any]:
        """
        给已有工单追加备注或用户补充信息。

        适用场景：
        - 用户补充了设备 SN、指示灯状态、网络环境等信息。
        - 用户补充新的故障现象。
        - 客服或工程师需要把处理过程写入工单。
        """

        try:
            ticket=add_ticket_note(
                ticket_id=ticket_id,
                note=note
            )
            return {
                "ok": True,
                "action_type": "追加备注",
                "ticket": ticket,
            }
        except Exception as e:
            return {
                'ok': False,
                'action_type': '追加备注',
                'error': str(e)
            }

    return [
        create_ticket_tool,
        get_ticket_detail_tool,
        update_ticket_status_tool,
        add_ticket_note_tool
    ]


















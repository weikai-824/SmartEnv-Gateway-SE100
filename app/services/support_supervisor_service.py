import json
import re
from typing import Any
from app.services.ticket_agent_service import run_ticket_agent
from app.services.rag_diag_agent_service import run_rag_diag_agent
from app.services.support_session_service import get_or_create_support_session,update_support_session
from langsmith import traceable

ticket_statuses = ["待处理", "待用户补充", "待工程师处理", "已解决", "已关闭", "已升级人工"]
intent_keywords = {
    "创建工单": [
        "创建工单", "创建一个工单", "帮我创建", "新建工单",
        "提交工单", "开工单", "建单", "帮我建单", "转人工", "报修"
    ],
    "查询工单": [
        "查询工单", "查工单", "查一下这个工单", "工单详情",
        "这个工单", "当前工单", "工单进度", "处理到哪", "进展"
    ],
    "追加备注": [
        "补充", "追加", "备注", "更新信息", "新增信息",
        "再说明", "联系人", "电话", "安装位置"
    ],
    "更新状态": [
        "更新状态", "修改状态", "改成", "改为", "设为",
        "标记为", "已解决", "已关闭"
    ],
}

#定义清理函数
def clean_text(text:str|None) ->str:
    return str(text or '').strip()
#判断关键词
def has_keyword(text:str,keywords:list[str]) ->bool:
    state=any(keyword in text for keyword in keywords)
    return state

#提取工单ID
def get_ticket_id(query:str,ticket_id:str|None) ->str|None:
    if ticket_id:
        ticket_id=clean_text(ticket_id)
        return ticket_id
    match=re.search(r"T\d{8}[A-Z0-9]{6}",query.upper())
    if match:
        ticket_id=match.group(0)
        return ticket_id
    else:
        return None

#提取目标工单状态
def get_target_status(query:str) ->str|None:
    for status in ticket_statuses:
        if status in query:
            return status
    return None

#判断用户意图
def infer_intent(query:str,ticket_id:str|None=None) ->str:
    current_ticket_id=get_ticket_id(query,ticket_id)
    target_status=get_target_status(query)
    #推断意图
    if has_keyword(query,intent_keywords['创建工单']):
        return '创建工单'
    if current_ticket_id and target_status and has_keyword(query,intent_keywords['更新状态']):
        return '更新状态'
    if current_ticket_id and has_keyword(query,intent_keywords['追加备注']):
        return '追加备注'
    if not current_ticket_id and has_keyword(query,intent_keywords['追加备注']):
        return "缺少工单的补充信息"
    if current_ticket_id and has_keyword(query,intent_keywords['查询工单']):
        return '查询工单'
    return 'RAG诊断'

#汇总工具调用
def collect_tool_calls(*results:dict[str,Any] | None) ->list[dict[str,Any]]:
    tool_calls=[]
    for result in results:
        if not result:
            continue
        agent_name="RagDiagAgent" if "diagnosis" in result else "TicketAgent"
        for tool_call in result.get('tool_calls',[]) or []:
            tool_name=tool_call.get('name')
            if tool_name:
                tool_calls.append({
                    'agent':agent_name,
                    'tool':tool_name
                })
    return tool_calls

#从工单结果中取字段
def get_ticket_detail_field(ticket_result:dict[str,Any]|None,field:str) ->Any:
    if not ticket_result:
        return None
    ticket=ticket_result.get('ticket')
    if isinstance(ticket,dict):
        return ticket.get(field)
    return None

#设置最终状态关闭下一环动作
terminal_ticket_statuses = {"已解决", "已关闭"}
#将语义处理干净
def build_ticket_next_action(ticket_result: dict[str, Any] | None) -> str:
    status = get_ticket_detail_field(ticket_result, "status")

    if status in terminal_ticket_statuses:
        return "当前工单已进入终态，暂无需继续补充信息；如出现新问题，请创建新工单。"

    return "等待用户继续补充信息或查询处理进展"

#压缩对外展示的工单信息
def slim_ticket_for_support(ticket: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ticket, dict):
        return None
    notes = ticket.get("notes") or []

    return {
        "ticket_id": ticket.get("ticket_id"),
        "title": ticket.get("title"),
        "problem_type": ticket.get("problem_type"),
        "priority": ticket.get("priority"),
        "status": ticket.get("status"),
        "summary": ticket.get("summary"),
        "missing_information": ticket.get("missing_information") or [],
        "notes": [{"note": note.get("note"),"created_at": note.get("created_at"),}
            for note in notes
            if isinstance(note, dict)
        ]}

#把 RAG 诊断结果整理成创建工单请求
def build_create_ticket_query(user_query:str,rag_result:dict[str,Any]) ->str:
    diagnosis=rag_result.get('diagnosis') or {}
    return f"""
    用户要求创建工单，请调用 create_ticket 工具。
    原始问题：
    {user_query}

    RAG 诊断回答：
    {rag_result.get("final_answer", "")}
    结构化诊断：
    {json.dumps(diagnosis, ensure_ascii=False)}

    要求：
    1. 只基于 RAG 诊断结果创建工单。
    2. 不要编造新的产品知识或维修结论。
    3. missing_information 必须按 list[str] 传递。
    4. 信息不足时，状态设为“待用户补充”。
    5. 明确需要售后时，状态可设为“待工程师处理”。
    """.strip()

#构造统一返回
def build_response(
    session_id: str,
    intent: str,
    stage: str,
    final_answer: str,
    next_action: str,
    rag_result: dict[str, Any] | None = None,
    ticket_result: dict[str, Any] | None = None,
    active_ticket_id: str | None = None,
) -> dict[str, Any]:

    tool_calls = collect_tool_calls(rag_result, ticket_result)
    ticket = ticket_result.get("ticket") if ticket_result else None

    return {
        "session_id": session_id,
        "intent": intent,
        "stage": stage,
        "final_answer": final_answer,
        "next_action": next_action,
        "active_ticket_id": active_ticket_id,
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "diagnosis": rag_result.get("diagnosis") if rag_result else None,
        "ticket": slim_ticket_for_support(ticket),
    }

#构造supervison主函数
@traceable(
    name='SupportSupervisor',
    run_type='chain',
    tags=['support','supervisor']
)
def run_support_supervisor(
    query: str,
    session_id: str | None = None,
    ticket_id: str | None = None,
    doc_id: str | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    max_context_chars: int = 6000,
    max_chunk_chars: int = 1200,
    min_rerank_score: float | None = None,
) -> dict[str, Any]:
    query = clean_text(query)
    if not query:
        raise ValueError('query不能为空')

    support_session = get_or_create_support_session(session_id=session_id)

    session_id=support_session['session_id']
    current_ticket_id=get_ticket_id(query,ticket_id) or support_session.get('active_ticket_id')
    intent=infer_intent(query,current_ticket_id)

    #加一个没有工单号的拦截分支
    if intent=='缺少工单的补充信息':
        stage = "缺少工单编号"
        final_answer = "当前没有正在处理的工单，无法直接补充备注。请先创建工单，或提供需要补充信息的工单编号。"
        next_action = "请提供工单编号，或先描述问题并创建工单"

        return build_response(
            session_id=session_id,
            stage=stage,
            intent=intent,
            final_answer=final_answer,
            next_action=next_action,
            active_ticket_id=None
        )

    #如果是工单类请求
    if intent in {"查询工单", "更新状态", "追加备注"}:
        ticket_result = run_ticket_agent(query, ticket_id=current_ticket_id)
        active_ticket_id=get_ticket_detail_field(ticket_result,field='ticket_id') or current_ticket_id
        stage=get_ticket_detail_field(ticket_result,field='status') or "工单处理中"

        update_support_session(
            session_id=session_id,
            last_intent=intent,
            active_ticket_id=active_ticket_id,
            stage=stage
        )
        return build_response(
            session_id=session_id,
            intent=intent,
            stage=stage,
            final_answer=ticket_result.get('final_answer','工单操作已完成'),
            next_action=build_ticket_next_action(ticket_result),
            ticket_result=ticket_result,
            active_ticket_id=active_ticket_id,
        )

    #如果是RAG诊断后创建工单
    rag_result = run_rag_diag_agent(
        query=query,
        doc_id=doc_id,
        candidate_k=candidate_k,
        top_k=top_k,
        rrf_k=rrf_k,
        max_chunk_chars=max_chunk_chars,
        max_context_chars=max_context_chars,
        min_rerank_score=min_rerank_score
    )

    if intent == '创建工单':
        create_query = build_create_ticket_query(query, rag_result)

        ticket_result = run_ticket_agent(create_query)

        stage=get_ticket_detail_field(ticket_result,'status') or '已创建工单'
        active_ticket_id=get_ticket_detail_field(ticket_result,'ticket_id')
        final_answer=(
            f"{rag_result.get('final_answer','')}\n\n"
            f"{ticket_result.get('final_answer','工单已创建')}"
        ).strip()

        update_support_session(
            session_id=session_id,
            active_ticket_id=active_ticket_id,
            last_intent=intent,
            stage=stage,
            last_diagnosis=rag_result.get('diagnosis')
        )
        return build_response(
            session_id=session_id,
            intent=intent,
            stage=stage,
            final_answer=final_answer,
            next_action="等待工程师处理，或继续补充 SN、指示灯状态、联网方式等信息",
            rag_result=rag_result,
            ticket_result=ticket_result,
            active_ticket_id=active_ticket_id,
        )
    #如果只是RAG诊断
    diagnosis = rag_result.get("diagnosis") or {}
    missing_information = diagnosis.get("missing_information") or []
    need_after_sales = bool(diagnosis.get("need_after_sales"))
    if not rag_result.get("has_context"):
        stage = "未命中知识库"
        next_action = "请补充更具体的设备现象，或上传相关技术文档后重新提问"
    elif need_after_sales:
        stage = "建议创建工单"
        next_action = "用户确认后，可以继续说“帮我创建工单”"
    elif missing_information:
        stage = "待用户补充"
        next_action = "请用户补充：" + "、".join(missing_information)
    else:
        stage = "诊断完成"
        next_action = "按建议步骤排查；如果问题仍未解决，可继续创建工单"

    update_support_session(
        session_id=session_id,
        active_ticket_id=current_ticket_id,
        last_intent=intent,
        stage=stage,
        last_diagnosis=diagnosis
    )
    return build_response(
        session_id=session_id,
        intent=intent,
        stage=stage,
        final_answer=rag_result.get("final_answer", ""),
        next_action=next_action,
        rag_result=rag_result,
        active_ticket_id=current_ticket_id
    )

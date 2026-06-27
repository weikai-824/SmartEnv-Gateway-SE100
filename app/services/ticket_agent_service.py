from app.services.json_output_service import extract_json_object
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.config.settings import Settings
from langchain.agents.structured_output import ToolStrategy
from app.services.ticket_tool import build_ticket_tools
from app.schemas.ticket import TicketAgentFinal
from functools import lru_cache
from langsmith import traceable

terminal_statuses = {"已解决", "已关闭"}
#1.创建系统提示词
ticket_agent_system_prompt="""
你是 SmartEnv-Gateway SE-100 技术支持系统里的工单处理 Agent。

职责边界：
1. 你只负责工单流程，不负责 RAG 诊断。
2. 你不能编造产品知识、故障原因、维修步骤或技术结论。
3. 用户要求创建工单时，必须调用 create_ticket。
4. 用户查询工单时，必须调用 get_ticket_detail。
5. 用户更新状态时，必须调用 update_ticket_status。
6. 用户补充信息时，必须调用 add_ticket_note。
7. 如果用户只是询问技术排查，请提示需要先走 RAG 诊断，不要自行诊断。
工单状态只能使用：
- 待处理
- 待用户补充
- 待工程师处理
- 已解决
- 已关闭
- 已升级人工
工单优先级只能使用：
- P0
- P1
- P2
- P3
优先级参考：
- P0：严重影响核心业务、大面积不可用、明确紧急。
- P1：设备完全不可用、持续离线、云端连接失败、数据完全不上报。
- P2：一般故障，需要售后跟进。
- P3：咨询、轻微问题、记录性补充。
创建工单时：
- title 要简洁概括问题。
- problem_type 可以使用：供电问题、联网问题、云端连接问题、平台离线/绑定问题、数据不上报问题、本地配置页问题、OTA升级问题、告警问题、外接传感器问题、恢复出厂设置问题、产品边界问题、信息不足。
- summary 写给客服或工程师看，要客观，不要编造。
- missing_information 必须作为列表传给 create_ticket，例如 ["SN", "指示灯状态", "联网方式"]。
- 不要把多个缺失信息拼成一个逗号分隔的字符串。
- 不要把一个完整的信息项拆碎，例如“是否刚修改过 Wi-Fi、MQTT 地址或设备密钥”应该作为一个完整列表项。
- 如果信息不足但用户明确要求创建工单，也可以创建工单，状态通常设为“待用户补充”。
最终必须返回结构化结果：
- final_answer：给用户看的简洁回复
- action：本次执行的工单动作
"""

#2.创建大语言模型
def build_agent_model():
    chat_llm=ChatOpenAI(
        model=Settings.llm_model,
        base_url=Settings.llm_base_url,
        api_key=Settings.llm_api_key,
        temperature=0,
        timeout=Settings.llm_timeout,
        max_tokens=400
    )
    return chat_llm

#3.把别的对象转成字典
def to_dict(data:Any) ->dict|None:
    if data is None:
        return None
    if hasattr(data,'model_dump'):
        return data.model_dump()
    if isinstance(data,dict):
        return data
    return None

#5.统计agent实际调用了哪些工具
def collect_tool_calls(messages:list[Any]) ->list[dict]:
    ignored_tool_names={"TicketAgentFinal"}
    tool_calls=[]
    for message in messages:
        for tool_call in getattr(message,'tool_calls',[]) or []:
            name=tool_call.get('name')
            if not name or name in ignored_tool_names:
                continue
            tool_calls.append({'name':name})
    return tool_calls

#6.从 Agent 最后一条消息里解析最终 JSON，并用 Pydantic 校验结构
def get_agent_final_output(messages:list[Any]) ->dict|None:
    if not messages:
        return None
    final_message=messages[-1]
    data=extract_json_object(getattr(final_message,'content',None))

    if not data:
        return None

    try:
        return TicketAgentFinal.model_validate(data).model_dump()
    except Exception:
        return None


#7.从工具返回结果里取最后一次工单数据
def find_latest_ticket(messages:list[Any]) ->dict|None:
    for message in reversed(messages):
        data=extract_json_object(getattr(message,'content',None))
        if not data:
            continue

        ticket = data.get('ticket')
        if isinstance(ticket, dict):
            return ticket
        
    return None

#8.提前缓存agent对象防止每次都加载
@lru_cache(maxsize=1)
def get_ticket_agent():
    return create_agent(
        model=build_agent_model(),
        tools=build_ticket_tools(),
        system_prompt=ticket_agent_system_prompt,
        response_format=ToolStrategy(TicketAgentFinal)
    )

#9.工单agent运行主逻辑
@traceable(
    name="TicketAgent",
    run_type="chain",
    tags=["agent", "ticket"]
)
def run_ticket_agent(query:str,ticket_id:str|None=None) ->dict:
    query=str(query or '').strip()
    if not query:
        raise ValueError('query不能为空')
    user_message=query
    if ticket_id:
        user_message=f'{query}\n\n已知工单ID：{ticket_id}'
    agent = get_ticket_agent()

    agent_states=agent.invoke({
        'messages':[{
            'role':'user',
            'content':user_message
        }]},
        config = {
        "run_name": "TicketAgentInvoke",
        "tags": ["ticket-agent", "tool-calling"],
        "metadata": {
            "ticket_id": ticket_id,
        },
    })

    messages=agent_states.get('messages',[])
    tool_calls=collect_tool_calls(messages)
    final_output=to_dict(agent_states.get('structured_response')) or get_agent_final_output(messages) or {}

    ticket = find_latest_ticket(messages)
    action = final_output.get("action")
    final_answer = str(final_output.get('final_answer') or '').strip()

    if not final_answer:
        if ticket:
            current_ticket_id = ticket.get("ticket_id") or "未知"
            status = ticket.get("status") or "未知"
            final_answer = f"工单操作已完成。当前工单编号：{current_ticket_id}，状态：{status}。"
        elif tool_calls:
            final_answer = (
                "工单 Agent 已调用工单工具，但没有拿到可用的工单数据。"
                "本次工单操作结果未确认，请稍后重试或检查工单工具返回格式。"
            )
        else:
            final_answer = (
                "工单 Agent 未返回有效结构化结果，也没有成功调用工单工具。"
                "本次工单操作未确认完成，请重新发起请求。"
            )

    # 如果工单已经终态，对外不再提示用户继续补充缺失信息
    if ticket and ticket.get("status") in terminal_statuses:
        ticket = dict(ticket)
        ticket["missing_information"] = []
        if isinstance(action, dict):
            action = dict(action)
            action["next_required_info"] = []

        # 查询终态工单时，不继续展示“下一步需要补充信息”
        if any(call.get("name") == "get_ticket_detail" for call in tool_calls):
            notes = ticket.get("notes") or []
            note_text = "；".join(
                str(note.get("note", "")).strip()
                for note in notes
                if str(note.get("note", "")).strip()
            )
            final_answer = (
                f"工单编号 {ticket.get('ticket_id')} 的处理进展如下：\n"
                f"- 标题：{ticket.get('title')}\n"
                f"- 类型：{ticket.get('problem_type')}\n"
                f"- 优先级：{ticket.get('priority')}\n"
                f"- 状态：{ticket.get('status')}\n"
                f"- 摘要：{ticket.get('summary')}"
            )
            if note_text:
                final_answer += f"\n- 用户补充信息：{note_text}"
            final_answer += "\n- 当前工单已进入终态，暂无需继续补充信息。"

    return {
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "action": action,
        "ticket": ticket,
    }






























import json
import time
from typing import Any
import gradio as gr
from app.services.support_supervisor_service import run_support_supervisor
from concurrent.futures import TimeoutError as FutureTimeoutError
from app.core.support_runtime import (
    SupportTimeoutError,
    submit_support_to_worker,
    get_support_timeout_seconds,
)
from app.core.support_load_guard import SupportBusyError, support_load_guard
APP_CSS = """
#main-title {
    padding: 18px 22px;
    border-radius: 14px;
    background: linear-gradient(90deg, #f8fafc, #fff7ed);
    border: 1px solid #e5e7eb;
    margin-bottom: 14px;
}

#main-title h1 {
    margin: 0;
    font-size: 26px;
}

#main-title p {
    margin: 6px 0 0 0;
    color: #64748b;
    font-size: 14px;
}

.support-panel {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 12px;
    background: #ffffff;
}

.status-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

button.primary-btn {
    height: 44px;
    font-weight: 700;
}

.footer-hint {
    color: #64748b;
    font-size: 13px;
}
"""

#1.把Python对象转成前端可读的JSON
def dumps(data:Any) ->str:
    return json.dumps(data,ensure_ascii=False,indent=2)

#2.将工单信息摘要整理成右侧摘要
def ticket_summary_md(ticket:dict[str,Any]|None) ->str:
    if not ticket:
        return '暂无关联工单'
    missing=ticket.get('missing_information') or []
    missing_text="、".join(missing) if missing else '无'
    return f"""
    当前工单
    
    - 工单ID：{ticket.get("ticket_id") or "无"}
    - 标题：{ticket.get("title") or "无"}
    - 状态：{ticket.get("status") or "无"}
    - 优先级：{ticket.get("priority") or "无"}
    - 问题类型：{ticket.get("problem_type") or "无"}
    - 缺失信息：{missing_text}
    
    摘要：
    {ticket.get('summary') or '无'}
    """.strip()

#3.发送问题然后调用后端Supervisor主链路
def ask(query:str,history:list,session_id:str,active_ticket_id:str):
    query=str(query or '').strip()
    history=history or []
    if not query:
        yield (
                "",
                history,
                session_id,
                active_ticket_id,
                session_id,
                active_ticket_id,
                "",
                "",
                "",
                "暂无关联工单。",
                "{}",
                "[]",
            )
        return
    #用户问题先进入到聊天框
    base_history=history + [{'role':'user','content':query}]
    #第一段，马上告诉前端，请求已收到
    received_history=base_history + [{'role':'assistant','content':"已接收请求，正在进入技术支持主链路..."}]
    yield (
        "",
        received_history,
        session_id,
        active_ticket_id,
        session_id,
        active_ticket_id,
        "",
        "已接收",
        "请求已进入技术支持主链路",
        "本轮请求处理中，暂无新的工单状态。",
        "{}",
        "[]",
    )
    time.sleep(0.5)
    #第二段，告诉前端，后端开始跑 RAG / Agent / LLM
    processing_history=base_history + [{'role':'assistant','content': "正在检索知识库、调用诊断 Agent 和工单 Agent，请稍候..."}]
    yield (
        "",
        processing_history,
        session_id,
        active_ticket_id,
        session_id,
        active_ticket_id,
        "",
        "处理中",
        "正在执行 RAG 检索、Agent 诊断和工单编排",
        "本轮请求处理中，暂无新的工单状态。",
        "{}",
        "[]",
    )
    try:
        with support_load_guard() as degraded:
            top_k = 5
            candidate_k = 20
            max_context_chars = 6000
            max_chunk_chars = 1200
            if degraded:
                top_k = 3
                candidate_k = 8
                max_context_chars = 3000
                max_chunk_chars = 800

            future=submit_support_to_worker(
                run_support_supervisor,
                query=query,
                session_id=session_id or None,
                ticket_id=active_ticket_id or None,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=60,
                max_context_chars=max_context_chars,
                max_chunk_chars=max_chunk_chars,
                min_rerank_score=None
            )
            timeout_seconds=get_support_timeout_seconds()
            start_time=time.monotonic()
            while True:
                try:
                    result=future.result(timeout=1)
                    break
                except FutureTimeoutError:
                    wait_time=int(time.monotonic() - start_time)
                    #先等一下，如果超时了再报错
                    if wait_time >= timeout_seconds:
                        future.cancel()
                        raise SupportTimeoutError("技术支持主链路处理超时，请稍后重试")
                    waiting_history=base_history + [
                    {
                        "role": "assistant",
                        "content": f"正在检索知识库、调用诊断 Agent 和工单 Agent，请稍候...（已等待 {wait_time} 秒）"
                    }
                ]
                    yield (
                        "",
                        waiting_history,
                        session_id,
                        active_ticket_id,
                        session_id,
                        active_ticket_id,
                        "",
                        "处理中",
                        f"正在执行 RAG 检索、Agent 诊断和工单编排，已等待 {wait_time} 秒",
                        "本轮请求处理中，暂无新的工单状态。",
                        "{}",
                        "[]",
                    )

            answer=result.get('final_answer') or '系统没有返回回答'
    except SupportTimeoutError as e:
        result={'error':str(e)}
        answer=f'请求失败：{e}'
    except Exception as e:
        result={'error':str(e)}
        answer=f'请求失败：{e}'
    # 第三段：主链路结束后，用最终回答替换“处理中”
    final_history = base_history + [
        {
            "role": "assistant",
            "content": answer
        }
    ]

    new_session_id=result.get('session_id') or session_id or ''
    new_ticket_id=result.get('active_ticket_id') or active_ticket_id or ''
    ticket=result.get('ticket') or {}
    intent=result.get("intent") or ""
    stage=result.get("stage") or ""
    next_action=result.get("next_action") or ""
    diagnosis=dumps(result.get("diagnosis"))
    tool_calls=dumps(result.get("tool_calls") or [])

    yield ("",
            final_history,
            new_session_id,
            new_ticket_id,
            new_session_id,
            new_ticket_id,
            intent,stage,
            next_action,
            ticket_summary_md(ticket),
            diagnosis,
            tool_calls)

#4.清空页面内容
def clear_content():
    return "", [], "", "", "", "", "", "", "", "暂无关联工单。", "{}", "[]"

#5.创建前端页面展示
def build_gradio_app() ->gr.Blocks:
    with gr.Blocks(
            title="SmartEnv 技术支持工作台",
            css=APP_CSS
    ) as demo:
        gr.HTML("""
        <div id="main-title">
            <h1>SmartEnv-Gateway SE-100 技术支持工作台</h1>
            <p>RAG 诊断 · Agent 工单编排 · 会话状态跟踪 · 售后处理闭环</p>
        </div>
        """)
        #创建两个隐藏状态用来存储对话和工单ID
        session_state=gr.State("")
        ticket_state=gr.State("")
        #布局一行两列和一行调试信息
        with gr.Row():
            with gr.Column(scale=2):
                chatbot=gr.Chatbot(label='技术支持对话',type='messages',height=500)
                query=gr.Textbox(label='用户问题',lines=3)
                with gr.Row():
                    send_btn = gr.Button("发送诊断 / 工单请求", variant="primary", elem_classes=["primary-btn"])
                    clear_btn = gr.Button("清空会话")
                gr.Examples(
                    examples=[
                        "SE-100 云端离线，重启也不行，帮我诊断一下。",
                        "SE-100 云端持续离线，重启后仍无法恢复，请帮我创建工单。",
                        "补充一下：设备 SN 是 SE100-DEMO-001，指示灯红灯常亮，联网方式是 Wi-Fi。",
                        "把这个工单状态改为待工程师处理。",
                        "查询这个工单的处理进展。",
                    ],
                    inputs=query
                )
            with gr.Column(scale=1):
                gr.Markdown("## 当前业务状态")
                gr.Markdown("展示 Supervisor 对本轮请求的意图识别、工单状态和下一步动作。")
                session_box = gr.Textbox(label="session_id", interactive=False)
                ticket_box = gr.Textbox(label="active_ticket_id", interactive=False)
                intent_box = gr.Textbox(label="intent", interactive=False)
                stage_box = gr.Textbox(label="stage", interactive=False)
                next_action_box = gr.Textbox(label="next_action", interactive=False)
                ticket_panel = gr.Markdown("暂无关联工单。")

        with gr.Accordion('调试信息',open=False):
            diagnosis_json=gr.Code(label='diagnosis',language='json')
            tool_calls_json=gr.Code(label='tool_calls',language='json')
        inputs = [query, chatbot, session_state, ticket_state]
        outputs = [
            query,
            chatbot,
            session_state,
            ticket_state,
            session_box,
            ticket_box,
            intent_box,
            stage_box,
            next_action_box,
            ticket_panel,
            diagnosis_json,
            tool_calls_json
        ]
        send_btn.click(ask,inputs=inputs,outputs=outputs)
        query.submit(ask,inputs=inputs,outputs=outputs)
        clear_btn.click(clear_content,inputs=[],outputs=outputs)
    demo.queue()
    return demo





















from fastapi import FastAPI
from app.config.settings import Settings
from app.api.routes_health import router as health_router
from app.api.routes_documents import router as documents_router
from app.api.routes_embeddings import router as embeddings_router
from app.api.routes_chat import router as chat_router
from app.api.routes_agent import router as agent_router
from app.api.routes_support import router as support_router
import gradio as gr
from app.ui.gradio_support_ui import build_gradio_app
from app.services.warmup_service import should_enable_startup_warmup, warmup_support_runtime

#创建FastAPI对象，后续所有API路由都要挂在这个对象上
def create_app() ->FastAPI:
    app=FastAPI(
        title=Settings.project_name,
        version='0.1.0',
        description="SmartEnv-Gateway SE-100 技术支持系统，支持 RAG 诊断、Agent 工具调用、会话状态管理和工单流程"
    )

    @app.on_event("startup")
    def startup_warmup() -> None:
        if should_enable_startup_warmup():
            warmup_support_runtime()

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(embeddings_router)
    app.include_router(chat_router)
    app.include_router(agent_router)
    app.include_router(support_router)

    support_ui=build_gradio_app()
    app=gr.mount_gradio_app(app,support_ui,'/ui')

    return app
app=create_app()
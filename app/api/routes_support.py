from fastapi import APIRouter, HTTPException
from app.services.support_supervisor_service import run_support_supervisor
from app.schemas.support import SupportSupervisorRequest,SupportSupervisorResponse
from app.core.support_load_guard import SupportBusyError, support_load_guard
from app.core.support_runtime import SupportTimeoutError, run_support_in_worker
#1.创建路由分组对象
router=APIRouter(prefix='/support',tags=['support'])

#2.注册post接口
@router.post('/ask',response_model=SupportSupervisorResponse)
async def support_ask_api(request: SupportSupervisorRequest):
    '''统一技术支持入口，返回一次完整业务结果'''
    try:
        with support_load_guard() as degraded:
            top_k = request.top_k
            candidate_k = request.candidate_k
            max_context_chars = request.max_context_chars
            max_chunk_chars = request.max_chunk_chars

            # 高并发压力情况下主动降级，降低召回数量和上下文长度
            if degraded:
                top_k = min(top_k, 3)
                candidate_k = min(candidate_k, 8)
                candidate_k = max(top_k, candidate_k)
                max_context_chars = min(max_context_chars, 3000)
                max_chunk_chars = min(max_chunk_chars, 800)

            result = await run_support_in_worker(
                run_support_supervisor,
                query=request.query,
                session_id=request.session_id,
                ticket_id=request.ticket_id,
                doc_id=request.doc_id,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=request.rrf_k,
                max_context_chars=max_context_chars,
                max_chunk_chars=max_chunk_chars,
                min_rerank_score=request.min_rerank_score,
            )

            return result

    except SupportBusyError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    except SupportTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"failed to run support supervisor: {e}"
        ) from e


















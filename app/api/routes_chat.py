from fastapi import APIRouter,HTTPException
from app.services.rag_service import answer_by_rag
from app.schemas.chat import ChatQueryRequest,ChatQueryResponse

#创建路由分组对象
router=APIRouter(prefix='/chat',tags=['chat'])

#注册post接口
@router.post('/query',response_model=ChatQueryResponse)
def chat_query_api(request:ChatQueryRequest):
    try:
        result=answer_by_rag(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            rrf_k=request.rrf_k,
            max_chunk_chars=request.max_chunk_chars,
            max_context_chars=request.max_context_chars,
            min_rerank_score=request.min_rerank_score
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to query chat:{e}') from e
































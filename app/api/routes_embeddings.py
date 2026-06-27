from fastapi import HTTPException,APIRouter
from app.services.embedding_service import embed_document_chunk
from app.schemas.embedding import EmbedDocumentRequest, EmbedDocumentResponse

#1.创建路由分组对象
router=APIRouter(prefix='/embeddings',tags=['embeddings'])
#注册接口
#对文档生成embedding
@router.post('/document',response_model=EmbedDocumentResponse)
def document_embedding_api(request:EmbedDocumentRequest):
    try:
        result=embed_document_chunk(doc_id=request.doc_id,batch_size=request.batch_size)
        return EmbedDocumentResponse(
            doc_id=result["doc_id"],
            embedding_dim=result["embedding_dim"],
            total_embeddings=result["total_embeddings"],
            source_total_chunks=result.get("source_total_chunks"),
            created_at=result["created_at"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=str(e))from e
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e))from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to embed document:{e}')from e













from fastapi import APIRouter,UploadFile,File,HTTPException
from app.services.index_service import index_document
from app.schemas.document import (
    ChunkDocumentRequest,
    ChunkDocumentResponse,
    DocumentIngestRequest,
    DocumentIndexRequest,
    DocumentIndexResponse,
    DocumentSummary,
    ListDocumentsResponse,
    ParsedDocumentResponse,
    UploadDocumentResponse,
)
from app.services.document_service import (
    get_ingest_document,
    ingest_document,
    list_uploaded_documents,
    save_uploaded_document,
)
from app.services.chunk_service import (
    chunk_document,
    get_document_chunk_results,
)
#1.创建路由分组对象
router=APIRouter(prefix='/documents',tags=['documents'])
#2.注册POST接口
#关于原文档的
@router.post('/upload',response_model=UploadDocumentResponse)
async def uploaded_document_api(file:UploadFile=File(...)):
    try:
        document=await save_uploaded_document(file)
        return UploadDocumentResponse(
            message='document uploaded successfully',
            document=DocumentSummary.from_meta(document)
        )
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to upload document:{e}')from e
@router.get('',response_model=ListDocumentsResponse)
def list_documents_api():
    documents = list_uploaded_documents()
    return ListDocumentsResponse(
        total=len(documents),
        documents=[DocumentSummary.from_meta(document) for document in documents]
    )
#关于解析后文档的
@router.post('/ingest',response_model=ParsedDocumentResponse)
def ingested_document_api(request:DocumentIngestRequest):
    try:
        doc_id=request.doc_id
        parsed_document=ingest_document(doc_id)
        return parsed_document
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to ingest document:{e}') from e

@router.get('/{doc_id}',response_model=ParsedDocumentResponse)
def get_ingested_document_api(doc_id: str):
    document=get_ingest_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404,detail=f'document is not ingested: {doc_id}')
    return document

#关于切分后文档的
@router.post('/chunk',response_model=ChunkDocumentResponse)
def document_chunk_api(request:ChunkDocumentRequest):
    try:
        result=chunk_document(doc_id=request.doc_id,chunk_size=request.chunk_size,chunk_overlap=request.chunk_overlap)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to chunk document:{e}') from e
@router.get('/{doc_id}/chunks',response_model=ChunkDocumentResponse)
def get_document_chunk_api(doc_id: str):
    results=get_document_chunk_results(doc_id)
    if results is None:
        raise HTTPException(status_code=404,detail=f'document is not found:{doc_id}')
    return results

@router.post('/index',response_model=DocumentIndexResponse)
def index_document_api(request:DocumentIndexRequest):
    try:
        result=index_document(
            doc_id=request.doc_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            batch_size=request.batch_size,
            delete_old=request.delete_old
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404,detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'failed to index document: {e}') from e




















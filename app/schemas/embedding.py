'''负责定义 API 的请求体和响应体'''
# app/schemas/embedding.py

from pydantic import BaseModel, Field


class EmbedDocumentRequest(BaseModel):
    """文档 chunks 批量 embedding 请求体。"""
    doc_id: str = Field(..., min_length=1, description="文档 ID")
    batch_size: int = Field(default=8, ge=1, le=64, description="批处理大小")

class EmbedDocumentResponse(BaseModel):
    """文档 chunks 批量 embedding 响应体。"""
    doc_id: str
    embedding_dim: int
    total_embeddings: int
    source_total_chunks: int | None = None
    created_at: str
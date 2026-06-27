'''负责定义 chat API 的请求体和响应体'''

from typing import Any
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    '''知识库问答请求体'''
    query: str = Field(..., min_length=1, description='用户问题')
    doc_id: str | None = Field(default=None, description='可选，限制只在某个文档内检索')
    top_k: int = Field(default=5, ge=1, le=20, description='最终返回给大模型的 chunk 数量')
    candidate_k: int = Field(default=20, ge=1, le=100, description='rerank 前的候选 chunk 数量')
    rrf_k: int = Field(default=60, ge=1, le=200, description='RRF 融合参数')
    max_context_chars: int = Field(default=6000, ge=500, le=20000, description='最大上下文字符数')
    max_chunk_chars: int = Field(default=1200, ge=100, le=5000, description='单个 chunk 最大字符数')
    min_rerank_score: float | None = Field(
        default=None,
        description='rerank_score 最小阈值；为空时不过滤低相关结果'
    )


class ChatContext(BaseModel):
    '''单条引用上下文'''
    context_id: int
    citation_id: str | None = None
    chunk_id: str | None = None
    source: str | None = None
    text: str

class CitationCheck(BaseModel):
    '''校验并清理答案中的引用编号'''
    valid_citation_ids: list[str] = Field(default_factory=list)
    used_citation_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    is_valid: bool = True
    answer_was_cleaned: bool = False

class ChatQueryResponse(BaseModel):
    '''知识库问答响应体'''
    query: str
    query_plan: dict[str, Any] | None = None
    answer: str
    has_context: bool
    contexts: list[ChatContext] = Field(default_factory=list)
    citation_check: CitationCheck | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] | None = None
    retrieval_status: str | None = Field(default=None, description='检索状态：success / no_context')
    no_context_reason: str | None = Field(default=None, description='无上下文原因，例如 no_retrieval_hits')

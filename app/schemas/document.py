'''负责定义 API 的请求体和响应体'''
from pydantic import BaseModel, Field
from typing import Any
class DocumentMeta(BaseModel):
    '''表示一个上传文档的元数据，文档档案'''
    doc_id: str = Field(..., description="文档唯一 ID")
    file_name: str = Field(..., description="原始文件名")
    file_type: str = Field(..., description="文件类型，例如 txt、md、pdf、docx")
    file_path: str = Field(..., description="文件在本地保存后的路径")
    file_size: int = Field(..., description="文件大小，单位 byte")
    created_at: str = Field(..., description="上传时间")
    status: str = Field(default="uploaded", description="文档当前状态")

class DocumentSummary(BaseModel):
    '''对外展示的文档摘要，不暴露本地 file_path'''
    doc_id: str = Field(..., description="文档唯一 ID")
    file_name: str = Field(..., description="原始文件名")
    file_type: str = Field(..., description="文件类型，例如 txt、md、pdf、docx")
    file_size: int = Field(..., description="文件大小，单位 byte")
    created_at: str = Field(..., description="上传时间")
    status: str = Field(default="uploaded", description="文档当前状态")

    @classmethod
    def from_meta(cls, document: DocumentMeta | dict[str, Any]) -> "DocumentSummary":
        data = document.model_dump() if isinstance(document, DocumentMeta) else document

        return cls(
            doc_id=data["doc_id"],
            file_name=data["file_name"],
            file_type=data["file_type"],
            file_size=data["file_size"],
            created_at=data["created_at"],
            status=data.get("status", "uploaded"),
        )

class UploadDocumentResponse(BaseModel):
    '''表示上传接口的返回结构'''
    message: str
    document: DocumentSummary


class ListDocumentsResponse(BaseModel):
    '''返回当前系统里已经上传过的文档列表'''
    total: int
    documents: list[DocumentSummary]

class DocumentIngestRequest(BaseModel):
    '''表示文档解析请求'''
    doc_id: str = Field(..., description="需要解析的文档 ID")


class ParsedDocumentResponse(BaseModel):
    '''表示文档解析后的返回结构'''
    doc_id: str = Field(..., description="文档唯一 ID")
    file_name: str = Field(..., description="原始文件名")
    file_type: str = Field(..., description="文件类型")
    created_at: str = Field(..., description="上传时间")
    text: str = Field(..., description="解析后的纯文本内容")

class ChunkDocumentRequest(BaseModel):
    '''表示文档切分请求'''
    doc_id: str = Field(..., description='需要切分的文档 ID')
    chunk_size: int = Field(default=500, description='每个 chunk 的最大字符数')
    chunk_overlap: int = Field(default=100, description='相邻 chunk 的重叠字符数')


class DocumentChunk(BaseModel):
    '''表示单个 chunk'''
    chunk_id: str = Field(..., description='chunk 唯一 ID')
    doc_id: str = Field(..., description='所属文档 ID')
    chunk_index: int = Field(..., description='chunk 序号')
    text: str = Field(..., description='chunk 文本')
    char_start: int = Field(..., description='chunk 在原文中的起始字符位置')
    char_end: int = Field(..., description='chunk 在原文中的结束字符位置')
    file_name: str = Field(default="", description="来源文件名")
    source: str = Field(default="", description="来源描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="chunk 元数据")

class ChunkDocumentResponse(BaseModel):
    '''表示文档切分后的返回结构'''
    doc_id: str = Field(..., description='文档 ID')
    total_chunks: int = Field(..., description='chunk 总数')
    chunk_size: int = Field(..., description='切分长度')
    chunk_overlap: int = Field(..., description='重叠长度')
    created_at: str = Field(..., description='切分时间')
    chunks: list[DocumentChunk] = Field(..., description='chunk 列表')

class DocumentIndexRequest(BaseModel):
    '''表示单文档入库请求：从已上传文档收敛到 Milvus 可检索状态'''
    doc_id: str = Field(..., description='需要入库的文档 ID')
    chunk_size: int = Field(default=500, ge=100, le=3000, description='每个 chunk 的最大字符数')
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description='相邻 chunk 的重叠字符数')
    batch_size: int = Field(default=8, ge=1, le=64, description='embedding 批处理大小')
    delete_old: bool = Field(default=True, description='入库前是否删除该 doc_id 的旧向量，避免重复污染检索')


class DocumentIndexResponse(BaseModel):
    '''表示单文档入库结果'''
    doc_id: str
    status: str
    delete_old: bool
    file_name: str | None = None
    total_chunks: int
    inserted_count: int

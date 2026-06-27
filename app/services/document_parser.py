'''输入一条文档元数据记录，输出统一的解析结果'''
from pathlib import Path
from typing import Any
from docx import Document
from pypdf import PdfReader


#根据文档记录，解析成统一文本结构
def parse_document(record:dict[str,Any]) ->dict[str,Any]:
    file_path=Path(record['file_path'])
    file_type=_normalize_file_type(record)
    if not file_path.exists():
        raise FileNotFoundError(f'文件{file_path}不存在')
    if file_type in {'txt','md'}:
        text=_read_text_file(file_path)
    elif file_type=='pdf':
        text=_read_pdf_file(file_path)
    elif file_type=='docx':
        text=_read_docx_file(file_path)
    else:
        raise ValueError(f'暂不支持的文件类型{file_type}')
    return {
        'doc_id':record['doc_id'],
        "file_name": record.get("file_name", file_path.name),
        "file_type": file_type,
        "created_at": record.get("created_at", ""),
        'text':text
    }

#统一文本类型格式
def _normalize_file_type(record:dict[str,Any]):
    file_type=record.get('file_type')
    if not file_type:
        file_path=Path(record['file_path'])
        file_type=file_path.suffix
    return str(file_type).lower().replace('.','')
#读取txt,md文件
def _read_text_file(file_path:Path) -> str:
    text=file_path.read_text(encoding='utf-8',errors='ignore')
    return text
#读取PDF文件
def _read_pdf_file(file_path:Path) -> str:
    reader=PdfReader(str(file_path))
    pages_text:list[str]=[]
    for page_index,page in enumerate(reader.pages):
        page_text=page.extract_text() or ''
        page_text=page_text.strip()
        if page_text:
            pages_text.append(page_text)
    return '\n\n'.join(pages_text)
#读取docx文件
def _read_docx_file(file_path:Path) -> str:
    document=Document(str(file_path))
    paragraphs:list[str]=[]
    for paragraph in document.paragraphs:
        text=paragraph.text.strip()
        if text:
            paragraphs.append(text)
    return '\n\n'.join(paragraphs)










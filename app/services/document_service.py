'''负责保存文件和读写元数据'''
from pathlib import Path
from app.config.settings import Settings
from app.schemas.document import DocumentMeta
from fastapi import UploadFile
import uuid
from datetime import datetime
import json
from typing import Any
from app.services.document_parser import parse_document
#设置允许上传的文件类型
Allowed_File_Types={'txt','md','pdf','docx'}

#确认上传目录存在
def get_upload_dir() -> Path:
    #将字符串路径转为Path对象
    upload_dir=Path(Settings.upload_dir)
    upload_dir.mkdir(parents=True,exist_ok=True)
    return upload_dir
#获取元数据文件路径
#documents.jsonl 用来记录上传过哪些文档
def get_metadata_path() ->Path:
    processed_dir=Path(Settings.processed_data_dir)
    processed_dir.mkdir(parents=True,exist_ok=True)
    return processed_dir / "documents_metadata.jsonl"
#获取文件类型
def get_file_type(file_name:str)-> str:
    #获取后缀名，小写并去掉.
    return Path(file_name).suffix.lower().replace('.','')
#保存上传的文档
#输入上传文件对象，输出文档元数据对象
async def save_uploaded_document(file:UploadFile)->DocumentMeta:
    #1.获取上传文件名，并判断是否为空
    file_name=file.filename
    if file_name is None:
        raise ValueError('Filename is EMPTY')
    #2.获取文件类型，并判断是否是可允许上传的文件类型
    file_type=get_file_type(file_name)
    if file_type not in Allowed_File_Types:
        raise ValueError('File type is not support')
    #3.获取文件唯一Id并拼接保存后的文件名
    doc_id=uuid.uuid4().hex
    save_file_name=f'{doc_id}_{file_name}'
    #4.拼接保存后的路径
    upload_dir=get_upload_dir()
    file_path=upload_dir/save_file_name
    #5.读出上传文件内容并保存
    content=await file.read()
    file_path.write_bytes(content)
    #6.生成一条文档档案
    document=DocumentMeta(
        doc_id=doc_id,
        file_type=file_type,
        file_name=file_name,
        file_path=str(file_path),
        file_size=len(content),
        created_at=datetime.now().isoformat(timespec='seconds'),
        status='uploaded'
    )
    #7.将文档档案记录进元数据文件里
    metadata_path=get_metadata_path()
    with metadata_path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(document.model_dump(),ensure_ascii=False) + '\n')
    return document
#读取上传文档列表
def list_uploaded_documents() ->list[DocumentMeta]:
    metadata_path=get_metadata_path()
    if not metadata_path.exists():
        return []
    documents=[]
    with metadata_path.open('r',encoding='utf-8') as f:
        for line in f:
            datas=json.loads(line)
            documents.append(DocumentMeta(**datas))
        return documents
#获取解析以后的保存目录
def get_ingested_path() -> Path:
    process_dir=Path(Settings.processed_data_dir)
    ingested_path=process_dir/'ingested'
    ingested_path.mkdir(parents=True,exist_ok=True)
    return ingested_path
#根据docs_id查找文档元数据
def find_document_by_id(doc_id:str) -> DocumentMeta|None:
    metadata_path=get_metadata_path()
    if not metadata_path.exists():
        return None
    with metadata_path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            data=json.loads(line)
            if data['doc_id']==doc_id:
                return DocumentMeta(**data)
    return None
#解析文档并写入
def ingest_document(doc_id:str) ->dict[str,Any]:
    document=find_document_by_id(doc_id)
    if document is None:
        raise ValueError(f'文档{doc_id}不存在')
    record=document.model_dump()
    parsed_document=parse_document(record)
    ingested_path=get_ingested_path()
    save_ingest_path=ingested_path/f'{doc_id}.json'
    with save_ingest_path.open('w',encoding='utf-8') as f:
        json.dump(parsed_document,f,ensure_ascii=False,indent=2)
    return parsed_document

#通过doc_id读取对应的解析后的文档
def get_ingest_document(doc_id:str) ->dict[str,Any]|None:
    ingested_path=get_ingested_path()
    save_ingest_path=ingested_path/f'{doc_id}.json'
    if not save_ingest_path.exists():
        return None
    with save_ingest_path.open('r',encoding='utf-8') as f:
        return json.load(f)
















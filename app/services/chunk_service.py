import json
import re
from datetime import datetime
from typing import Any
from app.config.settings import Settings
from app.services.document_service import get_ingest_document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
#1.获取项目根目录
def get_dir_root() ->Path:
    return Path(__file__).resolve().parents[2]

#2.获得文本切分后的chunk保存目录
def get_chunks_path() ->Path:
    processed_dir=Path(Settings.processed_data_dir)
    if not processed_dir.is_absolute():
        processed_dir=get_dir_root()/processed_dir
    chunks_path=processed_dir/'chunks'
    chunks_path.mkdir(parents=True,exist_ok=True)
    return chunks_path

#3.判断一行文本是否像章节标题
def is_section_title(line:str) ->bool:
    line=str(line or '').strip()
    if not line or len(line) > 80:
        return False

    patterns=[
        r"^第[一二三四五六七八九十0-9]+[章节部分]",
        r"^[一二三四五六七八九十]+[、.．]\s*\S+",
        r"^\d+(\.\d+)*[、.．]?\s+\S+",
        r"^#{1,6}\s+\S+",
    ]
    return any(re.match(pattern,line) for pattern in patterns)

#4.先按章节标题切成 section，后续再在每个 section 内切 chunk
def split_text_to_sections(text:str) ->list[dict[str,str]]:
    sections=[]
    current_title='正文'
    buffer=[]

    for line in text.splitlines(keepends=True):
        clean_line=line.strip()
        if is_section_title(clean_line) and buffer:
            sections.append({
                'section_title':current_title,
                'section_text':''.join(buffer)
            })
            current_title=clean_line.lstrip('#').strip()
            buffer=[line]
            continue
        if is_section_title(clean_line):
            current_title=clean_line.lstrip('#').strip()
        buffer.append(line)
    if buffer:
        sections.append({
            'section_title': current_title,
            'section_text': ''.join(buffer)
        })

    return sections or [{
        'section_title': '正文',
        'section_text': text
    }]

#5.根据langchain内置方法递归字符切分文本
def split_text_by_langchain(text:str,chunk_size:int=500,chunk_overlap:int=100) ->list[dict[str,Any]]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be greater than 0')

    if chunk_overlap < 0:
        raise ValueError('chunk_overlap must be greater than or equal to 0')

    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n','\n','。','！','？',';','，',' ','']
    )

    chunks=[]
    search_start=0
    chunk_index=0
    for section in split_text_to_sections(text):
        section_title=section.get('section_title') or '正文'
        section_text=section.get('section_text') or ''

        for chunk_text in text_splitter.split_text(section_text):
            clean_text=chunk_text.strip()
            if not clean_text:
                continue

            char_start = text.find(clean_text, search_start)
            if char_start == -1:
                char_start = search_start
            char_end = char_start + len(clean_text)

            chunks.append({
                'chunk_index': chunk_index,
                'section_title': section_title,
                'text': clean_text,
                'char_start': char_start,
                'char_end': char_end
            })
            chunk_index += 1
            search_start = max(char_end - chunk_overlap, char_start + 1)

    return chunks

#6.根据doc_id对已解析过的文档进行划分，业务函数
def chunk_document(doc_id:str,chunk_size:int=500,chunk_overlap:int=100) ->dict[str,Any]:
    ingested_document=get_ingest_document(doc_id)
    if ingested_document is None:
        raise FileNotFoundError(f'document has not been ingested: {doc_id}')
    file_name=ingested_document.get('file_name','')
    file_type=ingested_document.get('file_type','')
    text=ingested_document.get('text','')
    if not text.strip():
        raise ValueError(f'document text is empty: {doc_id}')
    #切分
    raw_chunks=split_text_by_langchain(text=text,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    # 生成结构化数据
    chunks=[]
    for chunk in raw_chunks:
        chunk_id=f"{doc_id}_{chunk['chunk_index']}"
        chunks.append({
            'chunk_id':chunk_id,
            'doc_id':doc_id,
            'chunk_index':chunk['chunk_index'],
            'text':chunk['text'],
            'char_start': chunk['char_start'],
            'char_end': chunk['char_end'],
            'file_name':file_name,
            'source':f"{file_name}#chunk_{chunk['chunk_index']}",
            'metadata':{
                'file_type':file_type,
                'chunk_size':chunk_size,
                'chunk_overlap':chunk_overlap,
                'splitter':'RecursiveCharacterTextSplitter',
                'chunking_strategy': 'section_aware_recursive',
                'section_title': chunk.get('section_title', '正文')
            }
        })
    results={
        'doc_id':doc_id,
        'total_chunks':len(chunks),
        'chunk_size':chunk_size,
        'chunk_overlap':chunk_overlap,
        'created_at':datetime.now().isoformat(timespec='seconds'),
        'chunks':chunks
    }
    #保存chunks
    chunk_path=get_chunks_path()
    save_chunks_path=chunk_path/f'{doc_id}.json'
    with save_chunks_path.open('w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    return results

#7.按doc_id读取已经切分好的chunk
def get_document_chunk_results(doc_id:str) ->dict[str,Any]|None:
    chunk_path=get_chunks_path()
    save_chunks_path=chunk_path/f"{doc_id}.json"
    if not save_chunks_path.exists():
        return None
    with save_chunks_path.open('r',encoding='utf-8') as f:
        return json.load(f)

#8.读取所有文档里的chunks
def get_all_documents_chunk_results() ->list[dict[str,Any]]:
    chunk_path=get_chunks_path()
    chunk_files=sorted(chunk_path.glob('*.json'))
    chunk_results=[]
    for chunk_file in chunk_files:
        doc_id=chunk_file.stem
        chunk_result=get_document_chunk_results(doc_id)
        if not chunk_result:
            continue
        chunk_results.append(chunk_result)
    return chunk_results

















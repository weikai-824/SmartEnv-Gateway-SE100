import json
from pathlib import Path
from datetime import datetime
from typing import Any
from app.services.embedding_service import embed_document_chunk
from app.services.chunk_service import chunk_document,get_all_documents_chunk_results
from app.services.document_service import ingest_document
from app.db.milvus_client import (
    build_milvus_id,
    create_milvus_collection,
    delete_doc_chunks,
    insert_chunk_embeddings,
    count_doc_chunks
)
from app.db.elasticsearch_client import (
    create_es_chunks_index,
    delete_es_doc_chunks,
    bulk_inserted_chunk_to_es,
    count_es_doc_chunks
)

#索引状态目录
index_job_dir=Path('data/processed/index_jobs')
#1.返回当前时间字符串，方便记录索引状态更新时间
def _now_time():
    return datetime.now().isoformat(timespec='seconds')

#2.根据doc_id构建对应状态文件路径
def _get_index_job_path(doc_id:str) ->Path:
    safe_doc_id=doc_id.replace('/','_').replace('\\','_')
    return index_job_dir / f"{safe_doc_id}.json"

#3.保存文档索引状态到对应文件,状态机
def save_index_job_status(
        doc_id:str,
        status:str,
        message:str='',
        extra:dict[str,Any]|None=None
) ->None:
    index_job_dir.mkdir(parents=True,exist_ok=True)
    data={
        'doc_id':doc_id,
        'status':status,
        'message':message,
        'updated_at':_now_time()
    }
    if extra:
        data.update(extra)
    path=_get_index_job_path(doc_id)
    with path.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

#4.读取某个文档的索引状态
def load_index_job_status(doc_id:str) ->dict[str,Any]|None:
    path=_get_index_job_path(doc_id)
    if not path.exists():
        return None
    with path.open('r',encoding='utf-8') as f:
        return json.load(f)


#5.将向量化后的chunk转换成Milvus可以插入的entities
def build_entities_from_embed_result(embedding_results:dict[str,Any]) ->list[dict[str,Any]]:
    doc_id=str(embedding_results.get('doc_id') or '').strip()
    if not doc_id:
        raise ValueError('embedding_result缺少doc_id')
    embedding_chunks=embedding_results.get('embeddings',[])

    if not embedding_chunks:
        raise ValueError(f'embedding_result里没有embedding:{doc_id}')

    entities:list[dict[str,Any]]=[]
    for chunk in embedding_chunks:
        text=str(chunk.get('text') or '').strip()
        chunk_id=str(chunk.get('chunk_id') or '').strip()
        chunk_index=int(chunk.get('chunk_index',0))
        if not text:
            continue
        if not chunk_id:
            raise ValueError(f"chunk_id 为空: doc_id={doc_id}, chunk_index={chunk_index}")
        file_name = str(chunk.get("file_name") or "unknown")
        source = str(chunk.get("source") or f"{file_name}#chunk_{chunk_index}")
        metadata = chunk.get("metadata") or {}
        entity = {
            "id": build_milvus_id(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text,
                source=source,
                chunk_index=chunk_index,
            ),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "text": text,
            "char_start": int(chunk.get("char_start", 0)),
            "char_end": int(chunk.get("char_end", len(text))),
            "file_name": file_name,
            "source": source,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "embedding": chunk["embedding"],
        }
        entities.append(entity)
    if not entities:
        raise ValueError(f'没有可入库的entities:{doc_id}')
    return entities

#6.单文档入库主链路，文档上传后把已有的服务串起来
def index_document(
        doc_id:str,
        chunk_size:int=500,
        chunk_overlap:int=100,
        batch_size:int=8,
        delete_old:bool=True
)->dict[str,Any]:
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        raise ValueError("doc_id 不能为空")

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须大于等于 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    expected_chunk_count = 0
    try:
        create_milvus_collection(drop_old=False)
        create_es_chunks_index(drop_old=False)
        
        #在重新解析，切分，上传之前先检查一次，幂等重试
        last_job_status=load_index_job_status(doc_id)
        if last_job_status and last_job_status.get('status') != 'indexed':
            delete_doc_chunks(doc_id)
            delete_es_doc_chunks(doc_id)
            save_index_job_status(
                doc_id=doc_id,
                status='indexing',
                message="检测到上次索引未完成，已清理 Milvus 和 Elasticsearch 残留数据，准备重新索引",
                extra={
                    'last_status':last_job_status.get('status'),
                    'last_message':last_job_status.get('message')
                }
            )
        parsed_result = ingest_document(doc_id)
        chunk_document(
            chunk_overlap=chunk_overlap,
            doc_id=doc_id,
            chunk_size=chunk_size
        )
        embedding_results=embed_document_chunk(
            doc_id=doc_id,
            batch_size=batch_size
        )
        entities=build_entities_from_embed_result(embedding_results)
        expected_chunk_count=len(entities)
        #先保存状态
        save_index_job_status(
            doc_id=doc_id,
            status='indexing',
            message="开始写入 Milvus 和 Elasticsearch",
            extra={'expected_chunk_count':expected_chunk_count}
        )

        if delete_old:
            delete_doc_chunks(doc_id)
            delete_es_doc_chunks(doc_id)
        insert_chunk_embeddings(entities)
        bulk_inserted_chunk_to_es(chunks=entities)
        #拿到两个数据库实际插入的chunk数量再来判断,最终一致性校验
        milvus_count=count_doc_chunks(doc_id)
        es_count=count_es_doc_chunks(doc_id)
        if milvus_count!=expected_chunk_count or es_count!=expected_chunk_count:
            raise RuntimeError(
                "索引一致性校验失败: "
                f"expected_chunk_count={expected_chunk_count}, "
                f"milvus_count={milvus_count}, "
                f"es_count={es_count}"
            )

        save_index_job_status(
            doc_id=doc_id,
            status='indexed',
            message="Milvus 和 Elasticsearch 写入完成，并通过一致性校验",
            extra={
                'expected_chunk_count':expected_chunk_count,
                'milvus_count':milvus_count,
                'es_count':es_count,
                "consistency_status": "ok"
            }
        )
        return {
            "doc_id": doc_id,
            "status": "indexed",
            "file_name": parsed_result.get("file_name"),
            "indexed_chunk_count": expected_chunk_count,
            "milvus_count": milvus_count,
            "es_count": es_count,
            "consistency_status": "ok",
            }
    #失败补偿
    except Exception as e:
        try:
            delete_doc_chunks(doc_id)
            delete_es_doc_chunks(doc_id)
        except Exception as cleanup_e:
            print(f'错误{cleanup_e}')

        save_index_job_status(
            doc_id=doc_id,
            status='failed',
            message=f'索引写入失败:{e}',
            extra={'expected_chunk_count':expected_chunk_count}
        )
        raise














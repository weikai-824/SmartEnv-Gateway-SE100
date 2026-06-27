from elasticsearch import Elasticsearch,helpers
from app.config.settings import Settings
from typing import Any

#1.建立连接
def get_es_client() -> Elasticsearch:
    return Elasticsearch(hosts=Settings.es_url,request_timeout=Settings.es_timeout)

#2.创建倒排索引
def create_es_chunks_index(drop_old:bool=False) ->dict[str,Any]:
    client=get_es_client()
    index_name=Settings.es_index_name

    if client.indices.exists(index=index_name):
        if not drop_old:
            return {
                'create':False,
                'reason':'already exists'
            }
        client.indices.delete(index=index_name)

    client.indices.create(
        index=index_name,
        settings={
            'number_of_shards':1,
            'number_of_replicas':0
        },
        mappings={
            "dynamic": "false",
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "file_name": {"type": "keyword"},
                "source": {"type": "keyword"},
            },
        },
    )

    return {"created":True}

#3.把原有的Milvus entity转成es_docu
def build_es_chunk_doc(chunk:dict[str,Any]) ->dict[str,Any]:
    return {
        'doc_id':str(chunk.get('doc_id') or ''),
        'chunk_id':str(chunk.get('chunk_id') or ''),
        'chunk_index':int(chunk.get('chunk_index') or 0),
        'text':str(chunk.get('text') or ''),
        "char_start": int(chunk.get("char_start", 0)),
        "char_end": int(chunk.get("char_end", 0)),
        "file_name": str(chunk.get("file_name") or ""),
        "source": str(chunk.get("source") or "")
    }

#4.根据doc_id删除已经入库的旧数据
def delete_es_doc_chunks(doc_id:str) ->dict[str,Any]:
    client=get_es_client()
    index_name=Settings.es_index_name
    create_es_chunks_index(drop_old=False)
    result=client.delete_by_query(
        index=index_name,
        query={'term':{'doc_id':doc_id}},
        conflicts='proceed',
        refresh=True
    )
    return {'deleted_count':int(result.get('deleted',0))}

#5.根据doc_id查询es里已有的chunk数量
def count_es_doc_chunks(doc_id:str) ->int:
    doc_id=str(doc_id or '').strip()
    if not doc_id:
        return 0
    create_es_chunks_index(drop_old=False)
    #建立连接
    index_name=Settings.es_index_name
    client=get_es_client()
    
    result=client.count(
        index=index_name,
        query={
            'term':{
                'doc_id':doc_id
            }
        }
    )
    return int(result.get('count',0))
#6.批量写入es
def bulk_inserted_chunk_to_es(chunks:list[dict[str,Any]]) ->dict[str,Any]:
    if not chunks:
        return {'inserted_count':0}
    client=get_es_client()
    index_name=Settings.es_index_name
    create_es_chunks_index(drop_old=False)

    actions=[]
    for chunk in chunks:
        doc=build_es_chunk_doc(chunk)
        if not doc['doc_id'] or not doc['chunk_id'] or not doc['text']:
            continue
        es_id=f"{doc['doc_id']}_{doc['chunk_id']}"
        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_id": es_id,
            "_source": doc,
        })
    if not actions:
        return {'inserted_count':0}
    success_count,_=helpers.bulk(
        client=client,
        actions=actions,
        refresh=True
    )

    return {'inserted_count':int(success_count)}

#6.ES查询函数
def search_es_chunks(query:str,doc_id:str|None=None,top_k:int=5) ->dict[str,Any]:
    query=str(query or '').strip()
    if not query:
        raise ValueError('query不能为空')
    if top_k <= 0:
        raise ValueError('top_k必须大于0')
    client=get_es_client()
    index_name=Settings.es_index_name
    create_es_chunks_index(drop_old=False)
    es_query={
        'bool':{
            'must':[
                {
                    'match':{
                        'text':query
                    }
                }
            ]
        }
    }
    if doc_id:
        es_query['bool']['filter']=[
            {
                'term':{
                    'doc_id':doc_id
                }
            }
        ]
    result=client.search(
        index=index_name,
        query=es_query,
        size=top_k
    )
    raw_hits=result.body.get('hits',{}).get('hits',[])
    hits=[]
    for rank,raw_hits in enumerate(raw_hits,start=1):
        source=raw_hits.get('_source',{})
        hits.append({
            'rank':rank,
            'sparse_score':float(raw_hits.get('_score') or 0.0),
            "doc_id": source.get("doc_id"),
            "chunk_id": source.get("chunk_id"),
            "chunk_index": source.get("chunk_index"),
            "text": source.get("text"),
            "char_start": source.get("char_start"),
            "char_end": source.get("char_end"),
            "file_name": source.get("file_name"),
            "source": source.get("source"),
        })
    return {
        'total_hits':len(hits),
        'hits':hits
    }


















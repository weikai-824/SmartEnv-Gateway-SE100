from pymilvus import MilvusClient,DataType

from app.config.settings import Settings
from typing import Any
import hashlib
#全局配置
collection_name=Settings.milvus_collection_name
embedding_dim=Settings.milvus_vector_dim
milvus_url=Settings.milvus_url
#0. 生成稳定主键id,输出128位的16进制字符串
def build_milvus_id(doc_id:str,chunk_id:str,text:str,source:str,chunk_index:int)->str:
    cleaned_text=text.strip().replace('\r\n','\n')
    #长文本缩短成短字符串
    text_hash=hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
    #拼接id
    raw=f'{doc_id}_{chunk_id}_{text_hash}_{source}_{chunk_index}'
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

#1.连接Milvus数据库并返回MilvusClient对象
def get_milvus_client(milvus_url) -> MilvusClient:
    client=MilvusClient(uri=milvus_url)
    return client

#2.创建并加载collection
def create_milvus_collection(drop_old:bool=False):
    client=get_milvus_client(milvus_url)
    #如果原本就有collection的话
    if client.has_collection(collection_name):
        if drop_old:
            client.drop_collection(collection_name)
        else:
            client.load_collection(collection_name)
            return
    #如果没有的话就创建一个
    #创建表结构
    schema=client.create_schema(auto_id=False,enable_dynamic_field=False)
    schema.add_field(field_name='id',datatype=DataType.VARCHAR,is_primary=True,max_length=128)
    schema.add_field(field_name='chunk_id',datatype=DataType.VARCHAR,max_length=256)
    schema.add_field(field_name='doc_id',datatype=DataType.VARCHAR,max_length=128)
    schema.add_field(field_name='chunk_index',datatype=DataType.INT64)
    schema.add_field(field_name='text',datatype=DataType.VARCHAR,max_length=8192)
    schema.add_field(field_name='char_start',datatype=DataType.INT64)
    schema.add_field(field_name='char_end',datatype=DataType.INT64)
    schema.add_field(field_name='file_name',datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name='source',datatype=DataType.VARCHAR,max_length=1024)
    schema.add_field(field_name='metadata_json',datatype=DataType.VARCHAR,max_length=4096)
    schema.add_field(field_name='embedding',datatype=DataType.FLOAT_VECTOR,dim=embedding_dim)
    #创建索引参数
    index_params=client.prepare_index_params()
    index_params.add_index(
        field_name='embedding',
        index_type='AUTOINDEX',
        metric_type=Settings.milvus_metric_type
    )
    client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)
    client.load_collection(collection_name=collection_name)

#3.根据doc_id删除旧的chunk，避免重复入库
def delete_doc_chunks(doc_id:str):
    client=get_milvus_client(milvus_url)
    client.load_collection(collection_name=collection_name)
    #取消转义，保留安全的字符串
    safe_doc_id=doc_id.replace('\\','\\\\').replace('"','\\"')
    client.delete(collection_name=collection_name,filter=f'doc_id=="{safe_doc_id}"')
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)

#4.统计某个doc_id在Milvus里的实际存在chunk数量
def count_doc_chunks(doc_id:str) ->int:
    doc_id=str(doc_id or '').strip()
    if not doc_id:
        return 0
    client=get_milvus_client(milvus_url)
    if not client.has_collection(collection_name):
        return 0
    client.load_collection(collection_name)
    safe_doc_id=doc_id.replace('\\','\\\\').replace('"','\\"')
    result=client.query(
        collection_name=collection_name,
        filter=f'doc_id=="{safe_doc_id}"',
        output_fields=['chunk_id']
    )
    return len(result)

#5.插入向量数据
def insert_chunk_embeddings(entities:list[dict[str,Any]]) ->dict[str,Any]:
    if not entities:
        return {'insert_count':0}
    client=get_milvus_client(milvus_url)
    result=client.insert(collection_name=collection_name,data=entities)
    client.flush(collection_name)
    client.load_collection(collection_name)
    if isinstance(result,dict):
        insert_count=int(result.get('insert_count',len(entities)))
    else:
        insert_count=len(entities)
    return {'insert_count':insert_count}

#6.封装Milvus搜索函数
def search_chunk_embeddings(query_embedding:list[float],doc_id:str|None=None,top_k:int=5) ->dict[str,Any]:
    if not query_embedding:
        raise ValueError('query_embedding不能为空')
    client=get_milvus_client(milvus_url)
    client.load_collection(collection_name=collection_name)
    #做转义
    filter_params=None
    if doc_id:
        safe_doc_id=doc_id.replace('\\','\\\\').replace('"','\\"')
        filter_params=f'doc_id=="{safe_doc_id}"'
    results=client.search(
        collection_name=collection_name,
        data=[query_embedding],
        anns_field='embedding',
        filter=filter_params,
        limit=top_k,
        output_fields=[
            'chunk_id',
            'doc_id',
            'chunk_index',
            'text',
            'char_start',
            'char_end',
            'file_name',
            'source'
        ],
        search_params={'metric_type':Settings.milvus_metric_type,
                       'params':{}}
    )
    hits=results[0] if results else []
    format_hits=[]
    for rank,hit in enumerate(hits,start=1):
        entity=hit.get('entity',{})
        one_hit={
            'rank':rank,
            'dense_score': hit.get('distance'),
            'doc_id':entity.get('doc_id'),
            'chunk_id':entity.get('chunk_id'),
            'chunk_index':entity.get('chunk_index'),
            'text':entity.get('text'),
            'char_start':entity.get('char_start'),
            'char_end':entity.get('char_end'),
            'file_name':entity.get('file_name'),
            'source':entity.get('source')
        }
        format_hits.append(one_hit)
    return {
        'total_hits':len(format_hits),
        'hits':format_hits
    }


















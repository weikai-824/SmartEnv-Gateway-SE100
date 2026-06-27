'''负责生成文本向量和保存chunk向量'''
import json
from pathlib import Path
from datetime import datetime
from app.config.settings import settings
from sentence_transformers import SentenceTransformer
from typing import Any
from app.services import chunk_service

embedding_dim = settings.milvus_vector_dim
#全局模型对象，避免每次请求都要重新加载
_embedding_model=None
#获取embedding保存目录
def get_embeddings_path() ->Path:
    process_dir=Path(settings.processed_data_dir)
    embeddings_path=process_dir/'embeddings'
    embeddings_path.mkdir(parents=True,exist_ok=True)
    return embeddings_path
#加载embedding模型
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_path = settings.embedding_model_path
        if not model_path:
            raise ValueError("EMBEDDING_MODEL_PATH 未配置，请检查项目根目录下的 .env 文件")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Embedding 模型路径不存在: {model_path}")

        _embedding_model = SentenceTransformer(
            model_path,
            device=settings.embedding_device,
        )
    return _embedding_model
#对单条文本生成embedding向量(用户问题)
def embed_text(text:str) ->list[float]:
    if not text:
        raise ValueError('text不能为空')
    embedding_model=get_embedding_model()
    embedding=embedding_model.encode(text,normalize_embeddings=True).tolist()
    if len(embedding) != embedding_dim:
        raise ValueError(f'embedding维度错误，期望{embedding_dim},实际{len(embedding)}')
    return embedding
#对一个文档里的所有chunk生成embedding并保存
def embed_document_chunk(doc_id:str,batch_size:int=8) ->dict[str,Any]:
    chunk_result=chunk_service.get_document_chunk_results(doc_id)
    if chunk_result is None:
        raise FileNotFoundError(f'document has not been chunk:{doc_id}')
    chunks=chunk_result.get('chunks',[])
    if not chunks:
        raise ValueError(f'document chunks is empty:{doc_id}')
    texts=[chunk['text'] for chunk in chunks]
    embedding_model=get_embedding_model()
    #将chunk向量化
    embeddings=embedding_model.encode(texts,batch_size=batch_size,
                                      normalize_embeddings=True,show_progress_bar=True).tolist()
    # 生成结构化数据
    embedding_chunks=[]
    for chunk,embedding in zip(chunks,embeddings):
        if len(embedding) != embedding_dim:
            raise ValueError(f'embedding维度错误，期望{embedding_dim},实际{len(embedding)}')
        embedding_chunks.append({
            'chunk_id':chunk.get('chunk_id'),
            'doc_id':chunk.get('doc_id'),
            'chunk_index':chunk['chunk_index'],
            'text':chunk['text'],
            'char_start': chunk['char_start'],
            'char_end': chunk['char_end'],
            'file_name':chunk['file_name'],
            'source':chunk['source'],
            'metadata':chunk['metadata'],
            'embedding':embedding
        })
    results={
        'doc_id':doc_id,
        'embedding_dim':embedding_dim,
        'total_embeddings':len(embedding_chunks),
        'source_total_chunks':chunk_result.get('total_chunks'),
        'created_at':datetime.now().isoformat(timespec='seconds'),
        'embeddings':embedding_chunks
    }
    #保存embeddings
    embeddings_path=get_embeddings_path()
    saved_embeddings_path=embeddings_path/f'{doc_id}.json'
    with saved_embeddings_path.open('w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    return results

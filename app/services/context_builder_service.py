from typing import Any
from langsmith import traceable

#将命中的hit转换为大模型能看懂的context最后放进prompt
@traceable(
    name="BuildRAGContext",
    run_type="chain",
    tags=["rag", "context-builder"]
)
def build_context_from_hits(hits:list[dict[str,Any]],
                            max_context_chars:int=6000,
                            max_chunk_chars:int=1200)->dict[str,Any]:
    if max_context_chars <=0:
        raise ValueError('max_context_chars必须大于零')
    if max_chunk_chars<=0:
        raise ValueError('max_chunk_chars必须大于零')
    contexts=[]
    context_text_parts=[]
    current_chars=0
    for idx,hit in enumerate(hits,start=1):
        text=str(hit.get('text') or '').strip()
        if not text:
            continue
        if len(text) > max_chunk_chars:
            text=text[:max_chunk_chars] + '...'
        source=hit.get('source') or hit.get('file_name') or 'unknown'
        chunk_id=hit.get('chunk_id')
        citation_id=f'C{idx}'
        one_context_text = (
            f'引用ID:[{citation_id}]\n'
            f'以下内容如被使用，必须标注：[{citation_id}]\n'
            f'来源:{source}\n'
            f'chunk_id:{chunk_id}\n'
            f'内容:{text}\n'
        )
        if len(one_context_text) + current_chars > max_context_chars:
            break
        context_text_parts.append(one_context_text)
        current_chars+=len(one_context_text)
        contexts.append(
            {
                "context_id": idx,
                "citation_id": citation_id,
                "chunk_id": chunk_id,
                "source": source,
                "text": text,
            }
        )
    return {
            'context_text':'\n'.join(context_text_parts),
            'contexts':contexts,
            'total_contexts':len(contexts),
        }













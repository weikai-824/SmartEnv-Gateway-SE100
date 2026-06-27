from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.config.settings import Settings
from langsmith import traceable

#全局模型对象，避免每次请求都要重新加载
_llm_chain=None
#1.拼接链
def get_llm_chain():
    global _llm_chain
    if _llm_chain==None:
        chat_prompt=ChatPromptTemplate.from_messages([
            ('system','''
            你是一位严谨的智能硬件产品技术支持与故障排查助手。
            你只能根据“已知内容”回答用户问题，不能使用模型自己的知识补充事实。
            回答规则：
            1. 如果“已知内容”不足以回答问题，直接回答：当前知识库没有足够信息回答这个问题。
            2. 如果可以回答，必须基于已知内容作答，不能补充已知内容没有明确支持的原因、结论或推断。
            3. 关键结论后必须标注引用编号，例如 [C1]、[C2]。
            4. 引用编号只能来自“已知内容”中每个片段开头的 [C数字] 标记，不能编造不存在的编号。
            5. 不要把章节号、步骤号、FAQ 编号、工单编号当作引用编号。
            6. 不要输出“根据已知内容”“参考文献”“参考资料”这类空泛或论文式表达，直接回答问题。
            7. 回答要清晰、简洁，优先分点说明。
            8. 不要从条件语句做反向推理。例如“发生告警时会立即上报”不等于“没有告警就不会上报”。
            9. 如果不同片段之间存在冲突，要说明冲突，而不是强行合并。
            已知内容:
            {context}
            '''),
            ('human','用户问题:{question}')
        ])
        chat_llm=ChatOpenAI(
            model=Settings.llm_model,
            api_key=Settings.llm_api_key,
            base_url=Settings.llm_base_url,
            temperature=0,
            timeout=Settings.llm_timeout,
            max_tokens=500
        )
        output_parser=StrOutputParser()
        _llm_chain=chat_prompt | chat_llm |output_parser
    return _llm_chain

#2.调用链来回答问题
@traceable(
    name="GenerateRAGAnswer",
    run_type="llm",
    tags=["rag", "answer-generation"]
)
def generate_answer(query:str,context_text:str='') ->dict[str,Any]:
    query=query.strip()
    context_text=str(context_text or '').strip()
    if not query:
        raise ValueError('query不能为空')
    if not context_text:
        return {
            "answer": "当前知识库没有足够信息回答这个问题。",
            "model": Settings.llm_model,
        }
    rag_chain=get_llm_chain()
    answer=rag_chain.invoke({
        'context':context_text,
        'question':query
    })
    answer=str(answer).strip()
    return {
        "answer": answer,
        "model": Settings.llm_model,
    }













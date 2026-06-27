from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config.settings import Settings
from langsmith import traceable
from app.services.json_output_service import extract_json_object
#全局链对象，避免每次请求都要加载
_query_planner_chain=None

fast_fault_keywords = [
    "离线", "云端", "重启", "无法恢复", "连不上", "联网",
    "MQTT", "指示灯", "不上报", "OTA", "告警", "传感器"
]
planner_noise_phrases = [
    "请帮我创建工单",
    "帮我创建工单",
    "创建工单",
    "开工单",
    "提交工单",
    "转人工",
    "报修",
]

#1.Query Planner失败时直接用原始query做检索
def build_fallback_query_plan(query:str,reason:str='fallback') ->dict[str,Any]:
    return {
        "planning_status": "fallback",
        "reason": reason,
        "original_query": query,
        "intent": "普通问答",
        "normalized_query": query,
        "retrieval_query": query,
        "key_terms": [],
        "evidence_requirements": [],
        "missing_information": [],
    }

#2.创建Query Planner Chain
def get_query_planner_chain():
    global _query_planner_chain
    if _query_planner_chain == None:
        #创建提示词
        chat_prompt=ChatPromptTemplate.from_messages([
            ('system','''
            你是 RAG 系统中的 Query Planner，只负责把用户问题改写成更适合检索的形式，不负责回答问题。

            请只输出 JSON，不要输出 Markdown，不要输出解释。
            
            字段要求：
            - intent：用户意图，只能从以下中文标签中选择一个：故障排查 / 配置指导 / 概念解释 / 售后咨询 / 普通问答
            - normalized_query：更清晰的用户问题
            - retrieval_query：真正适合向量检索和 BM25 检索的查询语句
            - key_terms：关键词列表
            - evidence_requirements：后续 RAG 需要从知识库中查找的证据
            - missing_information：用户没有提供但排查可能需要的信息
            
            输出格式：
            {{
              "intent": "故障排查",
              "normalized_query": "更清晰的问题",
              "retrieval_query": "适合检索的问题",
              "key_terms": ["关键词1", "关键词2"],
              "evidence_requirements": ["需要查找的证据1"],
              "missing_information": ["缺失信息1"]
            }}
            '''),
            ('human','用户问题:{query}')
        ])
        chat_llm=ChatOpenAI(
            model=Settings.llm_model,
            base_url=Settings.llm_base_url,
            api_key=Settings.llm_api_key,
            timeout=Settings.llm_timeout,
            temperature=0,
            max_tokens=500
        )
        output_parser=StrOutputParser()
        _query_planner_chain=chat_prompt | chat_llm | output_parser

    return _query_planner_chain

#3.短问题走规则清洗即可不用非得调用模型
def should_use_fast_query_plan(query: str) -> bool:
    query = str(query or "").strip()
    if len(query) > 80:
        return False
    return any(keyword in query for keyword in fast_fault_keywords)

def build_fast_query_plan(query: str) -> dict[str, Any]:
    retrieval_query = query

    for phrase in planner_noise_phrases:
        retrieval_query = retrieval_query.replace(phrase, "")

    retrieval_query = retrieval_query.strip(" ，。；;")

    return {
        "planning_status": "fast_rule",
        "reason": "skip_llm_planner_for_short_fault_query",
        "original_query": query,
        "intent": "故障排查",
        "normalized_query": retrieval_query or query,
        "retrieval_query": retrieval_query or query,
        "key_terms": [
            keyword for keyword in fast_fault_keywords
            if keyword in query
        ],
        "evidence_requirements": [],
        "missing_information": [],
    }

#4.生成query_plan
@traceable(
    name="QueryPlanner",
    run_type="chain",
    tags=["rag", "query-planner"]
)
def plan_query(query:str) ->dict[str,Any]:
    query=query.strip()
    if not query:
        raise ValueError('query不能为空')
    #短句跳过大模型
    if should_use_fast_query_plan(query):
        return build_fast_query_plan(query)

    try:
        chain=get_query_planner_chain()
        raw_plan=chain.invoke({'query':query})
        plan=extract_json_object(raw_plan)
        if not plan:
            return build_fallback_query_plan(query=query,reason='planner_output_not_json')
        retrieval_query=str(plan.get('retrieval_query') or '').strip()
        normalized_query=str(plan.get('normalized_query') or '').strip()
        if not retrieval_query:
            retrieval_query=normalized_query or query
        return {
            "planning_status": "success",
            "reason": "planner_success",
            "original_query": query,
            "intent": str(plan.get("intent") or "普通问答").strip(),
            "normalized_query": normalized_query or query,
            "retrieval_query": retrieval_query,
            "key_terms": plan.get("key_terms", []),
            "evidence_requirements": plan.get("evidence_requirements", []),
            "missing_information": plan.get("missing_information", []),
        }
    except Exception as e:
        return build_fallback_query_plan(
            query=query,
            reason=f"planner_exception:{type(e).__name__}"
        )

#5.从query_plan中取出真正用于检索的query
def get_retrieval_query(query_plan:dict[str,Any],fallback_query:str) ->str:
    retrieval_query=str(query_plan.get('retrieval_query') or '').strip()
    return retrieval_query or fallback_query











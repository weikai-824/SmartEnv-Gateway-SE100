import json
import re
from typing import Any

#从大模型输出中提取json对象
def extract_json_object(content:Any) ->dict[str,Any] | None:
    #1.如果已经是字典
    if isinstance(content,dict):
        return content
    if hasattr(content,'model_dump'):
        data=content.model_dump()
        return data if isinstance(data,dict) else None
    if not isinstance(content,str):
        return None
    #2.如果是字符串，清理模型输出文本
    text = content.strip()

    # 兼容模型输出 ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        text = match.group(1).strip()

    # 兼容模型在 JSON 前后输出少量解释文字，只要两个花括号里面的内容
    start=text.find("{")
    end=text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None
    #解析成字典
    try:
        data=json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None

    return data if isinstance(data,dict) else None

















import argparse
import json
import os
import sys
from typing import Any

import requests


def print_json(title: str, data: Any) -> None:
    print(f"\n========== {title} ==========")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.post(url, json=payload, timeout=120)

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    print_json(f"POST {path} [{response.status_code}]", data)

    assert_true(
        response.status_code == 200,
        f"接口 {path} 返回状态码不是 200，实际是 {response.status_code}",
    )

    return data


def get_json(base_url: str, path: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(url, timeout=30)

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    print_json(f"GET {path} [{response.status_code}]", data)

    assert_true(
        response.status_code == 200,
        f"接口 {path} 返回状态码不是 200，实际是 {response.status_code}",
    )

    return data


def get_tool_names(result: dict[str, Any]) -> list[str]:
    tool_names = []

    for tool_call in result.get("tool_calls", []) or []:
        tool_name = tool_call.get("tool") or tool_call.get("name")
        if tool_name:
            tool_names.append(tool_name)

    return tool_names


def get_tool_agents(result: dict[str, Any]) -> list[str]:
    return [
        tool_call.get("agent")
        for tool_call in result.get("tool_calls", [])
        if tool_call.get("agent")
    ]


def build_support_payload(query: str, doc_id: str | None = None, ticket_id: str | None = None) -> dict[str, Any]:
    payload = {
        "query": query,
        "top_k": 5,
        "candidate_k": 20,
        "rrf_k": 60,
        "max_context_chars": 6000,
        "max_chunk_chars": 1200,
        "min_rerank_score": None,
    }

    if doc_id:
        payload["doc_id"] = doc_id

    if ticket_id:
        payload["ticket_id"] = ticket_id

    return payload


def test_rag_diagnose(base_url: str, doc_id: str | None = None) -> dict[str, Any]:
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="SE-100 云端离线，重启也不行，帮我诊断一下。",
            doc_id=doc_id,
        ),
    )

    tool_names = get_tool_names(result)
    tool_agents = get_tool_agents(result)

    assert_true(result.get("intent") == "RAG诊断", "普通技术问题应该被识别为 RAG诊断")
    assert_true("RagDiagAgent" in tool_agents, "RAG诊断场景应该调用 RagDiagAgent")
    assert_true("rag_diagnose" in tool_names, "RAG诊断场景应该调用 rag_diagnose 工具")
    assert_true(result.get("final_answer"), "RAG诊断场景 final_answer 不应为空")

    return result


def test_create_ticket(base_url: str, doc_id: str | None = None) -> dict[str, Any]:
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="SE-100 云端持续离线，重启后仍无法恢复，请帮我创建工单。",
            doc_id=doc_id,
        ),
    )

    tool_names = get_tool_names(result)
    tool_agents = get_tool_agents(result)

    assert_true(result.get("intent") == "创建工单", "创建工单请求应该被识别为 创建工单")
    assert_true("RagDiagAgent" in tool_agents, "创建工单前应该先调用 RagDiagAgent")
    assert_true("TicketAgent" in tool_agents, "创建工单场景应该调用 TicketAgent")
    assert_true("rag_diagnose" in tool_names, "创建工单前应该调用 rag_diagnose")
    assert_true("create_ticket" in tool_names, "创建工单场景应该调用 create_ticket")

    ticket_id = result.get("active_ticket_id")
    ticket = result.get("ticket") or {}

    assert_true(ticket_id, "创建工单后 active_ticket_id 不应为空")
    assert_true(ticket.get("ticket_id") == ticket_id, "ticket.ticket_id 应该等于 active_ticket_id")
    assert_true(ticket.get("status"), "创建工单后 ticket.status 不应为空")

    return result


def test_add_note(base_url: str, ticket_id: str) -> dict[str, Any]:
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="补充一下：设备 SN 是 SE100-TEST-001，指示灯为红灯常亮，联网方式是 Wi-Fi。",
            ticket_id=ticket_id,
        ),
    )

    tool_names = get_tool_names(result)

    assert_true(result.get("intent") == "追加备注", "补充信息请求应该被识别为 追加备注")
    assert_true("add_ticket_note" in tool_names, "追加备注场景应该调用 add_ticket_note")
    assert_true(result.get("active_ticket_id") == ticket_id, "追加备注后 active_ticket_id 应该保持不变")

    ticket = result.get("ticket") or {}
    notes = ticket.get("notes") or []

    assert_true(len(notes) >= 1, "追加备注后 notes 不应为空")

    return result


def test_update_status(base_url: str, ticket_id: str) -> dict[str, Any]:
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="把这个工单状态改为待工程师处理。",
            ticket_id=ticket_id,
        ),
    )

    tool_names = get_tool_names(result)
    ticket = result.get("ticket") or {}

    assert_true(result.get("intent") == "更新状态", "更新状态请求应该被识别为 更新状态")
    assert_true("update_ticket_status" in tool_names, "更新状态场景应该调用 update_ticket_status")
    assert_true(ticket.get("status") == "待工程师处理", "工单状态应该变成 待工程师处理")

    return result


def test_query_ticket(base_url: str, ticket_id: str) -> dict[str, Any]:
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="查询这个工单的处理进展。",
            ticket_id=ticket_id,
        ),
    )

    tool_names = get_tool_names(result)
    ticket = result.get("ticket") or {}

    assert_true(result.get("intent") == "查询工单", "查询请求应该被识别为 查询工单")
    assert_true("get_ticket_detail" in tool_names, "查询工单场景应该调用 get_ticket_detail")
    assert_true(ticket.get("ticket_id") == ticket_id, "查询返回的 ticket_id 应该正确")
    assert_true(ticket.get("status") == "待工程师处理", "查询返回的状态应该保持为 待工程师处理")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--doc-id", default=None)
    args = parser.parse_args()

    base_url = args.base_url
    doc_id = args.doc_id

    print(f"BASE_URL = {base_url}")
    print(f"DOC_ID = {doc_id}")

    get_json(base_url, "/health")

    test_rag_diagnose(base_url, doc_id=doc_id)

    create_result = test_create_ticket(base_url, doc_id=doc_id)
    ticket_id = create_result["active_ticket_id"]

    test_add_note(base_url, ticket_id=ticket_id)
    test_update_status(base_url, ticket_id=ticket_id)
    test_query_ticket(base_url, ticket_id=ticket_id)

    print("\n========== E2E SUPPORT SUPERVISOR PASSED ==========")
    print(f"ticket_id = {ticket_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n========== E2E SUPPORT SUPERVISOR FAILED ==========")
        print(str(e))
        sys.exit(1)
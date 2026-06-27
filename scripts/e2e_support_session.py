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


def get_tool_names(result: dict[str, Any]) -> list[str]:
    tool_names = []

    for tool_call in result.get("tool_calls", []) or []:
        tool_name = tool_call.get("tool") or tool_call.get("name")
        if tool_name:
            tool_names.append(tool_name)

    return tool_names


def build_support_payload(
    query: str,
    session_id: str | None = None,
    ticket_id: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "top_k": 5,
        "candidate_k": 20,
        "rrf_k": 60,
        "max_context_chars": 6000,
        "max_chunk_chars": 1200,
        "min_rerank_score": None,
    }

    if session_id:
        payload["session_id"] = session_id

    if ticket_id:
        payload["ticket_id"] = ticket_id

    if doc_id:
        payload["doc_id"] = doc_id

    return payload


def test_create_ticket_and_bind_session(base_url: str, doc_id: str | None = None) -> dict[str, Any]:
    """
    第一步：创建工单。

    这一步不传 session_id，让系统自动生成。
    创建成功后，系统应该返回：
    1. session_id
    2. active_ticket_id
    3. ticket.ticket_id
    """
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="SE-100 云端持续离线，重启后仍无法恢复，请帮我创建工单。",
            doc_id=doc_id,
        ),
    )

    tool_names = get_tool_names(result)
    session_id = result.get("session_id")
    active_ticket_id = result.get("active_ticket_id")
    ticket = result.get("ticket") or {}

    assert_true(result.get("intent") == "创建工单", "应该识别为 创建工单")
    assert_true(session_id, "创建工单后 session_id 不应为空")
    assert_true(active_ticket_id, "创建工单后 active_ticket_id 不应为空")
    assert_true(ticket.get("ticket_id") == active_ticket_id, "ticket.ticket_id 应该等于 active_ticket_id")

    assert_true("rag_diagnose" in tool_names, "创建工单前应该调用 rag_diagnose")
    assert_true("create_ticket" in tool_names, "创建工单时应该调用 create_ticket")

    return result


def test_add_note_by_session_only(
    base_url: str,
    session_id: str,
    ticket_id: str,
) -> dict[str, Any]:
    """
    第二步：只传 session_id，不传 ticket_id，追加备注。

    这是本脚本最关键的测试。
    如果这一步通过，说明 Supervisor 能从 support_sessions 表里恢复 active_ticket_id。
    """
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="补充一下：设备 SN 是 SE100-SESSION-001，指示灯红灯常亮，联网方式是 Wi-Fi。",
            session_id=session_id,
            ticket_id=None,
        ),
    )

    tool_names = get_tool_names(result)
    ticket = result.get("ticket") or {}
    notes = ticket.get("notes") or []

    assert_true(result.get("intent") == "追加备注", "只传 session_id 时，补充信息应该识别为 追加备注")
    assert_true("add_ticket_note" in tool_names, "追加备注场景应该调用 add_ticket_note")
    assert_true(result.get("active_ticket_id") == ticket_id, "active_ticket_id 应该从 session 中恢复")
    assert_true(ticket.get("ticket_id") == ticket_id, "备注应该追加到原工单")
    assert_true(len(notes) >= 1, "追加备注后 notes 不应为空")

    notes_text = json.dumps(notes, ensure_ascii=False)
    assert_true(
        "SE100-SESSION-001" in notes_text,
        "备注中应该包含本次补充的 SN 信息",
    )

    return result


def test_update_status_by_session_only(
    base_url: str,
    session_id: str,
    ticket_id: str,
) -> dict[str, Any]:
    """
    第三步：只传 session_id，不传 ticket_id，更新工单状态。
    """
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="把这个工单状态改为已解决。",
            session_id=session_id,
            ticket_id=None,
        ),
    )

    tool_names = get_tool_names(result)
    ticket = result.get("ticket") or {}

    assert_true(result.get("intent") == "更新状态", "只传 session_id 时，应该识别为 更新状态")
    assert_true("update_ticket_status" in tool_names, "更新状态场景应该调用 update_ticket_status")
    assert_true(result.get("active_ticket_id") == ticket_id, "active_ticket_id 应该保持为原工单")
    assert_true(ticket.get("ticket_id") == ticket_id, "更新状态应该作用到原工单")
    assert_true(ticket.get("status") == "已解决", "工单状态应该更新为 已解决")

    return result


def test_query_ticket_by_session_only(
    base_url: str,
    session_id: str,
    ticket_id: str,
) -> dict[str, Any]:
    """
    第四步：只传 session_id，不传 ticket_id，查询工单。
    """
    result = post_json(
        base_url=base_url,
        path="/support/ask",
        payload=build_support_payload(
            query="查询这个工单的处理进展。",
            session_id=session_id,
            ticket_id=None,
        ),
    )

    tool_names = get_tool_names(result)
    ticket = result.get("ticket") or {}

    assert_true(result.get("intent") == "查询工单", "只传 session_id 时，应该识别为 查询工单")
    assert_true("get_ticket_detail" in tool_names, "查询工单场景应该调用 get_ticket_detail")
    assert_true(result.get("active_ticket_id") == ticket_id, "active_ticket_id 应该从 session 中恢复")
    assert_true(ticket.get("ticket_id") == ticket_id, "查询结果应该是原工单")
    assert_true(ticket.get("status") == "已解决", "查询到的状态应该保持为 已解决")

    notes = ticket.get("notes") or []
    notes_text = json.dumps(notes, ensure_ascii=False)

    assert_true(
        "SE100-SESSION-001" in notes_text,
        "查询工单时应该能看到之前通过 session 追加的备注",
    )

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

    create_result = test_create_ticket_and_bind_session(base_url, doc_id=doc_id)

    session_id = create_result["session_id"]
    ticket_id = create_result["active_ticket_id"]

    test_add_note_by_session_only(
        base_url=base_url,
        session_id=session_id,
        ticket_id=ticket_id,
    )

    test_update_status_by_session_only(
        base_url=base_url,
        session_id=session_id,
        ticket_id=ticket_id,
    )

    test_query_ticket_by_session_only(
        base_url=base_url,
        session_id=session_id,
        ticket_id=ticket_id,
    )

    print("\n========== E2E SUPPORT SESSION PASSED ==========")
    print(f"session_id = {session_id}")
    print(f"ticket_id = {ticket_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n========== E2E SUPPORT SESSION FAILED ==========")
        print(str(e))
        sys.exit(1)
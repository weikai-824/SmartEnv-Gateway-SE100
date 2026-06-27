import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_EVAL_SET_PATH = "data/eval/support_e2e_eval_set.jsonl"
DEFAULT_OUTPUT_PATH = "data/eval/support_e2e_eval_results.json"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"评估集不存在：{eval_path}")

    cases: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败：{e}") from e

    return cases


def post_support_ask(
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/support/ask"
    response = requests.post(url, json=payload, timeout=timeout)

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    return response.status_code, data


def build_payload(
    query: str,
    session_id: str | None = None,
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

    if doc_id:
        payload["doc_id"] = doc_id

    return payload


def get_tool_names(result: dict[str, Any]) -> list[str]:
    tool_names: list[str] = []

    for tool_call in result.get("tool_calls", []) or []:
        tool_name = tool_call.get("tool") or tool_call.get("name")
        if tool_name:
            tool_names.append(tool_name)

    return tool_names


def get_tool_agents(result: dict[str, Any]) -> list[str]:
    agents: list[str] = []

    for tool_call in result.get("tool_calls", []) or []:
        agent = tool_call.get("agent")
        if agent:
            agents.append(agent)

    return agents


def build_check_text(result: dict[str, Any]) -> str:
    """
    评估时不只看 final_answer。
    因为工单状态、工单编号、备注等信息可能在 ticket / stage / next_action 里。
    """
    parts = [
        result.get("intent"),
        result.get("stage"),
        result.get("final_answer"),
        result.get("next_action"),
        json.dumps(result.get("ticket"), ensure_ascii=False),
    ]

    return "\n".join(str(part or "") for part in parts)


def contains_any(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True

    return any(keyword in text for keyword in keywords)


def contains_none(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True

    return not any(keyword in text for keyword in keywords)


def check_case_result(
    case: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> dict[str, Any]:
    expect = case.get("expect") or {}

    all_tool_names: list[str] = []
    all_tool_agents: list[str] = []

    for result in turn_results:
        all_tool_names.extend(get_tool_names(result))
        all_tool_agents.extend(get_tool_agents(result))

    final_result = turn_results[-1] if turn_results else {}
    check_text = build_check_text(final_result)

    should_use_rag = expect.get("should_use_rag")
    should_use_ticket = expect.get("should_use_ticket")
    should_create_ticket = expect.get("should_create_ticket")
    active_ticket_expected = expect.get("active_ticket_expected")

    must_include_any = expect.get("must_include_any") or []
    must_not_include_any = expect.get("must_not_include_any") or []
    expected_final_intent = expect.get("expected_final_intent")
    expected_final_stage = expect.get("expected_final_stage")
    checks: dict[str, bool] = {}

    if should_use_rag is not None:
        checks["rag_route_ok"] = ("rag_diagnose" in all_tool_names) == bool(should_use_rag)

    if should_use_ticket is not None:
        checks["ticket_route_ok"] = ("TicketAgent" in all_tool_agents) == bool(should_use_ticket)

    if should_create_ticket is not None:
        checks["create_ticket_ok"] = ("create_ticket" in all_tool_names) == bool(should_create_ticket)

    if active_ticket_expected is not None:
        checks["active_ticket_ok"] = bool(final_result.get("active_ticket_id")) == bool(active_ticket_expected)
    if expected_final_intent is not None:
        checks["final_intent_ok"] = final_result.get("intent") == expected_final_intent

    if expected_final_stage is not None:
        checks["final_stage_ok"] = final_result.get("stage") == expected_final_stage
    checks["must_include_ok"] = contains_any(check_text, must_include_any)
    checks["must_not_include_ok"] = contains_none(check_text, must_not_include_any)

    overall_pass = all(checks.values())

    return {
        "case_id": case.get("case_id"),
        "case_name": case.get("case_name"),
        "passed": overall_pass,
        "checks": checks,
        "all_tool_names": all_tool_names,
        "all_tool_agents": all_tool_agents,
        "final_intent": final_result.get("intent"),
        "final_stage": final_result.get("stage"),
        "final_active_ticket_id": final_result.get("active_ticket_id"),
        "final_answer": final_result.get("final_answer"),
        "next_action": final_result.get("next_action"),
        "final_ticket": final_result.get("ticket"),
    }


def run_one_case(
    case: dict[str, Any],
    base_url: str,
    doc_id: str | None,
    timeout: int,
) -> dict[str, Any]:
    turns = case.get("turns") or []
    if not turns:
        return {
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "passed": False,
            "error": "case 中没有 turns",
        }

    session_id: str | None = None
    turn_results: list[dict[str, Any]] = []
    turn_errors: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(turns, start=1):
        query = turn.get("user")
        if not query:
            turn_errors.append({
                "turn_index": turn_index,
                "error": "turn.user 为空",
            })
            break

        payload = build_payload(
            query=query,
            session_id=session_id,
            doc_id=doc_id,
        )

        try:
            status_code, result = post_support_ask(
                base_url=base_url,
                payload=payload,
                timeout=timeout,
            )
        except Exception as e:
            turn_errors.append({
                "turn_index": turn_index,
                "query": query,
                "error": str(e),
            })
            break

        if status_code != 200:
            turn_errors.append({
                "turn_index": turn_index,
                "query": query,
                "status_code": status_code,
                "response": result,
            })
            break

        turn_results.append(result)

        # 关键：多轮评估默认只传 session_id，不主动传 ticket_id。
        # 这样才能评估 active_ticket_id 是否真的能从 session 里恢复。
        session_id = result.get("session_id") or session_id

    if turn_errors:
        return {
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "passed": False,
            "error": "执行过程中出现错误",
            "turn_errors": turn_errors,
            "turn_results": turn_results,
        }

    checked = check_case_result(case, turn_results)
    checked["turn_count"] = len(turn_results)
    checked["turn_results"] = turn_results

    return checked


def summarize_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(case_results)
    passed_cases = sum(1 for item in case_results if item.get("passed"))
    failed_cases = total_cases - passed_cases

    summary: dict[str, Any] = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round(passed_cases / total_cases, 4) if total_cases else 0.0,
    }

    metric_names = [
        "rag_route_ok",
        "ticket_route_ok",
        "create_ticket_ok",
        "active_ticket_ok",
        "final_intent_ok",
        "final_stage_ok",
        "must_include_ok",
        "must_not_include_ok",
    ]

    for metric in metric_names:
        values = [
            item.get("checks", {}).get(metric)
            for item in case_results
            if metric in item.get("checks", {})
        ]

        if values:
            summary[f"{metric}_rate"] = round(
                sum(1 for value in values if value) / len(values),
                4,
            )

    return summary


def save_results(output_path: str, data: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--case-id", default=None)
    args = parser.parse_args()

    cases = load_jsonl(args.eval_set)

    if args.case_id:
        cases = [case for case in cases if case.get("case_id") == args.case_id]
        if not cases:
            raise ValueError(f"没有找到指定 case_id：{args.case_id}")

    print(f"BASE_URL = {args.base_url}")
    print(f"EVAL_SET = {args.eval_set}")
    print(f"OUTPUT = {args.output}")
    print(f"DOC_ID = {args.doc_id}")
    print(f"TOTAL_CASES = {len(cases)}")

    case_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id")
        case_name = case.get("case_name")

        print(f"\n[{index}/{len(cases)}] running {case_id} - {case_name}")

        result = run_one_case(
            case=case,
            base_url=args.base_url,
            doc_id=args.doc_id,
            timeout=args.timeout,
        )

        case_results.append(result)

        status = "PASS" if result.get("passed") else "FAIL"
        print(f"[{status}] {case_id}")

        if not result.get("passed"):
            print(json.dumps({
                "case_id": result.get("case_id"),
                "case_name": result.get("case_name"),
                "checks": result.get("checks"),
                "error": result.get("error"),
                "final_intent": result.get("final_intent"),
                "final_stage": result.get("final_stage"),
                "all_tool_names": result.get("all_tool_names"),
                "all_tool_agents": result.get("all_tool_agents"),
            }, ensure_ascii=False, indent=2))

    summary = summarize_results(case_results)

    output_data = {
        "summary": summary,
        "case_results": case_results,
    }

    save_results(args.output, output_data)

    print("\n========== SUPPORT E2E EVAL SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
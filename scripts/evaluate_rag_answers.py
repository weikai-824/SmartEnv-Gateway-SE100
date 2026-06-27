"""
M8.4 RAG Answer Evaluation

回答评估目标：
- 是否命中预期上下文
- 是否包含必要信息
- 是否出现禁止性错误
- citation 是否合法
- 是否整体通过
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.services.rag_service import answer_by_rag


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "answer_eval_set.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "eval" / "answer_eval_results.json"


NEGATION_PREFIXES = (
    "不",
    "不是",
    "不能",
    "不可",
    "不可以",
    "不应",
    "不应该",
    "无需",
    "并非",
    "不要",
    "不能直接",
    "不负责",
)


def load_eval_set(eval_file: Path) -> list[dict[str, Any]]:
    if not eval_file.exists():
        raise FileNotFoundError(f"回答评估集不存在: {eval_file}")

    items: list[dict[str, Any]] = []

    with eval_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {e}") from e

            query = str(item.get("query", "")).strip()
            if not query:
                raise ValueError(f"第 {line_no} 行 query 不能为空")

            items.append(
                {
                    "line_no": line_no,
                    "query_id": item.get("query_id") or f"line_{line_no}",
                    "query": query,
                    "answer_type": item.get("answer_type"),
                    "should_have_context": bool(item.get("should_have_context", True)),
                    "expected_chunk_ids": item.get("expected_chunk_ids", []),
                    "expected_context_keywords_any": item.get("expected_context_keywords_any", []),
                    "must_include_any": item.get("must_include_any", []),
                    "must_not_include": item.get("must_not_include", []),
                    "note": item.get("note", ""),
                }
            )

    if not items:
        raise ValueError(f"回答评估集为空: {eval_file}")

    return items


def normalize_text(text: str) -> str:
    """
    轻量归一化：
    - 英文统一小写，避免 CLOUD/cloud、MQTT/mqtt 误判
    - 去掉多余空白
    """
    return "".join(str(text or "").lower().split())


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized_text = normalize_text(text)

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            return True

    return False


def is_negated_occurrence(text: str, start_index: int) -> bool:
    """
    判断某个禁止词命中位置前面是否存在明显否定前缀。

    例子：
    - “不可以直接控制工业设备” 不应该被判定为违规
    - “不一定是设备坏了” 不应该被判定为违规
    """
    prefix_window = text[max(0, start_index - 8): start_index]
    return any(prefix_window.endswith(prefix) for prefix in NEGATION_PREFIXES)


def check_must_include_any(
    answer: str,
    must_include_any: list[list[str]],
) -> tuple[bool, list[list[str]]]:
    """
    must_include_any 格式：
    [
        ["关键词A1", "关键词A2"],
        ["关键词B1", "关键词B2"]
    ]

    每一组里命中任意一个即可；所有组都命中才算通过。
    """
    missing_groups: list[list[str]] = []

    for group in must_include_any:
        if not isinstance(group, list) or not group:
            continue

        keywords = [str(x) for x in group]
        if not contains_any(answer, keywords):
            missing_groups.append(keywords)

    return len(missing_groups) == 0, missing_groups


def check_must_not_include(
    answer: str,
    must_not_include: list[str],
) -> tuple[bool, list[str]]:
    """
    检查答案中是否出现禁止性错误结论。

    注意：
    这里不能简单做 substring 判断。
    例如“不是设备坏了”包含“设备坏了”，但语义是正确否定。
    所以这里会忽略带明显否定前缀的命中。
    """
    bad_keywords: list[str] = []
    answer_text = str(answer or "")

    for keyword in must_not_include:
        keyword_text = str(keyword or "").strip()
        if not keyword_text:
            continue

        start = 0
        while True:
            match_index = answer_text.find(keyword_text, start)
            if match_index == -1:
                break

            if not is_negated_occurrence(answer_text, match_index):
                bad_keywords.append(keyword_text)
                break

            start = match_index + len(keyword_text)

    return len(bad_keywords) == 0, bad_keywords


def extract_visible_citation_ids(answer: str) -> tuple[list[str], list[str]]:
    """
    从最终可见 answer 中提取引用。

    返回：
    - visible_citation_ids: 合法形态的 C 类引用，例如 C1、C2
    - unsupported_citation_tokens: 类似 [Q15] 这种非 C 类引用
    """
    answer_text = str(answer or "")

    visible_citation_ids = re.findall(r"\[(C\d+)\]", answer_text)
    bracket_tokens = re.findall(r"\[([A-Za-z]+\d+)\]", answer_text)

    unsupported_citation_tokens = [
        f"[{token}]"
        for token in bracket_tokens
        if not re.fullmatch(r"C\d+", token)
    ]

    return visible_citation_ids, unsupported_citation_tokens


def check_visible_citations(
    answer: str,
    contexts: list[dict[str, Any]],
    has_context: bool,
    should_have_context: bool,
) -> tuple[bool, bool, bool, bool, list[str], list[str], list[str]]:
    """
    检查最终展示给用户的 answer 引用是否合格。
    不直接依赖 rag_service 里记录的原始 citation_check。
    """
    visible_citation_ids, unsupported_citation_tokens = extract_visible_citation_ids(answer)

    valid_citation_ids = {
        str(context.get("citation_id"))
        for context in contexts
        if context.get("citation_id")
    }

    invalid_visible_citation_ids = [
        citation_id
        for citation_id in visible_citation_ids
        if citation_id not in valid_citation_ids
    ]

    citation_id_valid_ok = len(invalid_visible_citation_ids) == 0
    citation_format_ok = len(unsupported_citation_tokens) == 0

    citation_used_ok = True
    if has_context and should_have_context:
        citation_used_ok = len(visible_citation_ids) > 0

    citation_ok = citation_id_valid_ok and citation_format_ok and citation_used_ok

    return (
        citation_ok,
        citation_id_valid_ok,
        citation_format_ok,
        citation_used_ok,
        visible_citation_ids,
        invalid_visible_citation_ids,
        unsupported_citation_tokens,
    )


def check_expected_context(
    contexts: list[dict[str, Any]],
    expected_chunk_ids: list[str],
) -> tuple[bool, list[str]]:
    """
    检查返回上下文是否命中预期 chunk_id。
    """
    if not expected_chunk_ids:
        return True, []

    expected_set = {str(x) for x in expected_chunk_ids}
    actual_chunk_ids = [
        str(context.get("chunk_id"))
        for context in contexts
        if context.get("chunk_id")
    ]

    matched = [chunk_id for chunk_id in actual_chunk_ids if chunk_id in expected_set]
    return len(matched) > 0, matched


def check_expected_context_keywords(
    contexts: list[dict[str, Any]],
    expected_context_keywords_any: list[list[str]],
) -> tuple[bool, list[list[str]]]:
    """
    检查返回上下文是否包含预期证据关键词。

    格式：
    [
        ["关键词A1", "关键词A2"],
        ["关键词B1", "关键词B2"]
    ]

    每组命中任意一个即可；所有组都命中才算通过。
    """
    if not expected_context_keywords_any:
        return False, []

    context_text = "\n".join(
        str(context.get("text") or context.get("text_preview") or "")
        for context in contexts
    )

    missing_groups: list[list[str]] = []

    for group in expected_context_keywords_any:
        if not isinstance(group, list) or not group:
            continue

        keywords = [str(x) for x in group]
        if not contains_any(context_text, keywords):
            missing_groups.append(keywords)

    return len(missing_groups) == 0, missing_groups


def evaluate_one_case(
    item: dict[str, Any],
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    min_rerank_score: float | None,
) -> dict[str, Any]:
    query = item["query"]

    try:
        rag_result = answer_by_rag(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            min_rerank_score=min_rerank_score,
        )

        answer = str(rag_result.get("answer", "")).strip()
        contexts = rag_result.get("contexts", [])
        has_context = bool(rag_result.get("has_context", False))
        should_have_context = bool(item.get("should_have_context", True))

        citation_check = rag_result.get("citation_check") or {}

        (
            citation_ok,
            citation_id_valid_ok,
            citation_format_ok,
            citation_used_ok,
            visible_citation_ids,
            invalid_visible_citation_ids,
            unsupported_citation_tokens,
        ) = check_visible_citations(
            answer=answer,
            contexts=contexts,
            has_context=has_context,
            should_have_context=should_have_context,
        )

        has_context_ok = has_context == should_have_context

        must_include_ok, missing_include_groups = check_must_include_any(
            answer=answer,
            must_include_any=item.get("must_include_any", []),
        )

        must_not_include_ok, bad_keywords = check_must_not_include(
            answer=answer,
            must_not_include=item.get("must_not_include", []),
        )

        expected_chunk_ok, matched_expected_chunk_ids = check_expected_context(
            contexts=contexts,
            expected_chunk_ids=item.get("expected_chunk_ids", []),
        )

        expected_keyword_ok, missing_context_keyword_groups = check_expected_context_keywords(
            contexts=contexts,
            expected_context_keywords_any=item.get("expected_context_keywords_any", []),
        )

        # 证据检查：优先精确 chunk_id；如果 chunk_id 因重切分变化，也允许上下文关键词兜底。
        expected_context_ok = expected_chunk_ok or expected_keyword_ok

        overall_pass = all(
            [
                has_context_ok,
                citation_ok,
                must_include_ok,
                must_not_include_ok,
                expected_context_ok,
            ]
        )

        return {
            "line_no": item["line_no"],
            "query_id": item["query_id"],
            "query": query,
            "answer_type": item.get("answer_type"),
            "overall_pass": overall_pass,
            "checks": {
                "has_context_ok": has_context_ok,
                "citation_ok": citation_ok,
                "citation_id_valid_ok": citation_id_valid_ok,
                "citation_format_ok": citation_format_ok,
                "citation_used_ok": citation_used_ok,
                "must_include_ok": must_include_ok,
                "must_not_include_ok": must_not_include_ok,
                "expected_context_ok": expected_context_ok,
            },
            "debug": {
                "has_context": has_context,
                "should_have_context": should_have_context,
                "missing_include_groups": missing_include_groups,
                "bad_keywords": bad_keywords,
                "expected_chunk_ids": item.get("expected_chunk_ids", []),
                "matched_expected_chunk_ids": matched_expected_chunk_ids,
                "expected_chunk_ok": expected_chunk_ok,
                "expected_context_keywords_any": item.get("expected_context_keywords_any", []),
                "expected_keyword_ok": expected_keyword_ok,
                "missing_context_keyword_groups": missing_context_keyword_groups,
                "visible_citation_ids": visible_citation_ids,
                "invalid_visible_citation_ids": invalid_visible_citation_ids,
                "unsupported_citation_tokens": unsupported_citation_tokens,
                # service 侧原始 citation 检查结果保留作诊断。
                "service_used_citation_ids": citation_check.get("used_citation_ids", []),
                "service_invalid_citation_ids": citation_check.get("invalid_citation_ids", []),
                "answer_was_cleaned": citation_check.get("answer_was_cleaned", False),
                "retrieval": rag_result.get("retrieval", {}),
            },
            "answer": answer,
            "contexts": [
                {
                    "citation_id": context.get("citation_id"),
                    "chunk_id": context.get("chunk_id"),
                    "source": context.get("source"),
                    "rerank_score": context.get("rerank_score"),
                    "rerank_rank": context.get("rerank_rank"),
                    "text_preview": str(context.get("text", ""))[:160],
                }
                for context in contexts
            ],
            "error": None,
        }

    except Exception as e:
        return {
            "line_no": item["line_no"],
            "query_id": item["query_id"],
            "query": query,
            "answer_type": item.get("answer_type"),
            "overall_pass": False,
            "checks": {
                "has_context_ok": False,
                "citation_ok": False,
                "citation_id_valid_ok": False,
                "citation_format_ok": False,
                "citation_used_ok": False,
                "must_include_ok": False,
                "must_not_include_ok": False,
                "expected_context_ok": False,
            },
            "debug": {},
            "answer": "",
            "contexts": [],
            "error": repr(e),
        }


def summarize_results(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    passed = sum(1 for item in details if item["overall_pass"])
    errors = sum(1 for item in details if item.get("error"))

    check_names = [
        "has_context_ok",
        "citation_ok",
        "citation_id_valid_ok",
        "citation_format_ok",
        "citation_used_ok",
        "must_include_ok",
        "must_not_include_ok",
        "expected_context_ok",
    ]

    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "error_count": errors,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }

    for check_name in check_names:
        ok_count = sum(1 for item in details if item["checks"].get(check_name))
        summary[f"{check_name}_rate"] = round(ok_count / total, 4) if total else 0.0

    by_answer_type: dict[str, dict[str, Any]] = {}

    for item in details:
        answer_type = item.get("answer_type") or "unknown"

        if answer_type not in by_answer_type:
            by_answer_type[answer_type] = {
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "pass_rate": 0.0,
            }

        by_answer_type[answer_type]["total_cases"] += 1

        if item.get("overall_pass"):
            by_answer_type[answer_type]["passed_cases"] += 1
        else:
            by_answer_type[answer_type]["failed_cases"] += 1

    for stats in by_answer_type.values():
        total_cases = stats["total_cases"]
        stats["pass_rate"] = round(stats["passed_cases"] / total_cases, 4) if total_cases else 0.0

    summary["by_answer_type"] = by_answer_type
    summary["failed_query_ids"] = [
        item["query_id"]
        for item in details
        if not item.get("overall_pass")
    ]

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== RAG Answer Evaluation Summary ===")
    print(f"total_cases: {summary['total_cases']}")
    print(f"passed_cases: {summary['passed_cases']}")
    print(f"failed_cases: {summary['failed_cases']}")
    print(f"error_count: {summary['error_count']}")
    print(f"pass_rate: {summary['pass_rate']:.4f}")
    print("-" * 50)
    print(f"has_context_ok_rate: {summary['has_context_ok_rate']:.4f}")
    print(f"citation_ok_rate: {summary['citation_ok_rate']:.4f}")
    print(f"citation_id_valid_ok_rate: {summary['citation_id_valid_ok_rate']:.4f}")
    print(f"citation_format_ok_rate: {summary['citation_format_ok_rate']:.4f}")
    print(f"citation_used_ok_rate: {summary['citation_used_ok_rate']:.4f}")
    print(f"must_include_ok_rate: {summary['must_include_ok_rate']:.4f}")
    print(f"must_not_include_ok_rate: {summary['must_not_include_ok_rate']:.4f}")
    print(f"expected_context_ok_rate: {summary['expected_context_ok_rate']:.4f}")
    print("-" * 50)
    print("by_answer_type:")

    for answer_type, stats in summary.get("by_answer_type", {}).items():
        print(
            f"  {answer_type}: "
            f"pass_rate={stats['pass_rate']:.4f}, "
            f"passed={stats['passed_cases']}, "
            f"total={stats['total_cases']}"
        )

    if summary.get("failed_query_ids"):
        print("-" * 50)
        print("failed_query_ids:")
        for query_id in summary["failed_query_ids"]:
            print(f"  - {query_id}")


def save_results(results: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--min-rerank-score", type=float, default=None)

    args = parser.parse_args()

    eval_items = load_eval_set(args.eval_file)

    details = []

    for idx, item in enumerate(eval_items, start=1):
        print(f"[answer_eval] {idx}/{len(eval_items)} {item['query'][:40]}", flush=True)

        detail = evaluate_one_case(
            item=item,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            min_rerank_score=args.min_rerank_score,
        )

        details.append(detail)

        status = "PASS" if detail["overall_pass"] else "FAIL"
        print(f"[answer_eval] {idx}/{len(eval_items)} {status}", flush=True)

    summary = summarize_results(details)

    results = {
        "eval_name": "rag_answer_eval",
        "eval_file": str(args.eval_file),
        "run_config": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "min_rerank_score": args.min_rerank_score,
        },
        "summary": summary,
        "details": details,
    }

    print_summary(summary)
    save_results(results, args.output_file)

    print(f"\n回答评估明细已保存: {args.output_file}")


if __name__ == "__main__":
    main()

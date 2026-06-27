"""
这个脚本只评估“证据有没有找对”，不评估大模型最终回答。

评估对象：
- dense：向量检索
- sparse：关键词检索 / BM25 类检索
- hybrid：dense + sparse 融合检索
- hybrid_rerank：融合检索后再 rerank

评估流程：
1. 从 data/eval/retrieval_eval_set.jsonl 读取评估样本
2. 每条样本包含 query 和人工标注的 positive_chunk_ids
3. 分别调用不同检索器
4. 计算 Hit@K、Recall@K、MRR@K
5. 输出 data/eval/retrieval_eval_results.json

指标解释：
- Hit@K：前 K 条结果里是否至少命中一个正确 chunk
- Recall@K：前 K 条结果找回了多少比例的正确 chunk
- MRR@K：第一个正确 chunk 排得越靠前，分数越高

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


# 保证从 scripts/ 目录运行时，也能正常导入 app 下的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.services.dense_retrieval_service import dense_retrieve_by_query
from app.services.sparse_retrieval_service import sparse_retrieve_by_query
from app.services.hybrid_retrieval_service import hybrid_retrieve_by_query
from app.services.rerank_service import hybrid_rerank_retrieve_by_query


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_set.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_results.json"

HIT_KS = [1, 3, 5]
RECALL_KS = [3, 5]
MRR_K = 5

RetrieveFn = Callable[[str, str | None, int], dict[str, Any]]
def hybrid_rerank_retrieve_for_eval(
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """给 evaluate_retrieval.py 用的 hybrid_rerank 包装函数。"""
    return hybrid_rerank_retrieve_by_query(
        query=query,
        doc_id=doc_id,
        top_k=top_k,
        candidate_k=20,
        rrf_k=60,
    )

def load_eval_set(eval_file: Path) -> list[dict[str, Any]]:
    """
    读取检索评估集。

    每一行格式：
    {
        "query": "用户问题",
        "positive_chunk_ids": ["正确chunk_id_1", "正确chunk_id_2"]
    }
    """
    if not eval_file.exists():
        raise FileNotFoundError(f"评估集不存在: {eval_file}")

    eval_items = []

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
            positive_chunk_ids = item.get("positive_chunk_ids", [])

            # 兼容单个 positive_chunk_id 的误写情况，但主格式仍然是 positive_chunk_ids
            if not positive_chunk_ids and item.get("positive_chunk_id"):
                positive_chunk_ids = [item["positive_chunk_id"]]

            if not query:
                raise ValueError(f"第 {line_no} 行 query 为空")

            if not isinstance(positive_chunk_ids, list) or not positive_chunk_ids:
                raise ValueError(
                    f"第 {line_no} 行 positive_chunk_ids 必须是非空 list"
                )

            eval_items.append(
                {
                    "line_no": line_no,
                    "query_id": item.get("query_id") or f"line_{line_no}",
                    "query": query,
                    "query_type": item.get("query_type", "unknown"),
                    "positive_chunk_ids": [str(x) for x in positive_chunk_ids],
                    "positive_doc_ids": [str(x) for x in item.get("positive_doc_ids", [])],
                    "expected_answer_keywords": [
                        str(x) for x in item.get("expected_answer_keywords", [])
                    ],
                    "note": item.get("note", ""),
                }
            )

    if not eval_items:
        raise ValueError(f"评估集为空: {eval_file}")

    return eval_items


def calc_one_query_metrics(
    hits: list[dict[str, Any]],
    positive_chunk_ids: list[str],
) -> dict[str, float]:
    """
    计算单条 query 的检索指标。

    hits:
        检索器返回的 top_k 结果列表。

    positive_chunk_ids:
        人工标注的正确 chunk_id 列表。
    """
    positive_set = set(positive_chunk_ids)
    retrieved_chunk_ids = [str(hit.get("chunk_id")) for hit in hits]

    metrics: dict[str, float] = {}

    # Hit@K：前 K 条里是否至少有一个正确 chunk
    for k in HIT_KS:
        top_k_ids = retrieved_chunk_ids[:k]
        is_hit = any(chunk_id in positive_set for chunk_id in top_k_ids)
        metrics[f"hit@{k}"] = 1.0 if is_hit else 0.0

    # Recall@K：前 K 条找回了多少比例的正确 chunk
    for k in RECALL_KS:
        top_k_ids = set(retrieved_chunk_ids[:k])
        hit_positive_count = len(positive_set & top_k_ids)
        metrics[f"recall@{k}"] = hit_positive_count / len(positive_set)

    # MRR@5：第一个正确 chunk 出现在第几名
    first_hit_rank: int | None = None

    for rank, chunk_id in enumerate(retrieved_chunk_ids[:MRR_K], start=1):
        if chunk_id in positive_set:
            first_hit_rank = rank
            break

    metrics[f"mrr@{MRR_K}"] = 1.0 / first_hit_rank if first_hit_rank else 0.0

    return metrics

def analyze_retrieval_hits(
    hits: list[dict[str, Any]],
    positive_chunk_ids: list[str],
) -> dict[str, Any]:
    """
    提取一条 query 的命中明细，方便人工查看 JSON 结果。

    metrics 负责算分；
    这个函数负责告诉你：
    - 命中了哪些正确 chunk
    - 第一个正确 chunk 排在第几名
    """
    positive_set = set(positive_chunk_ids)

    matched_positive_chunk_ids: list[str] = []
    first_hit_rank: int | None = None

    for rank, hit in enumerate(hits, start=1):
        chunk_id = str(hit.get("chunk_id"))

        if chunk_id in positive_set:
            matched_positive_chunk_ids.append(chunk_id)

            if first_hit_rank is None:
                first_hit_rank = rank

    return {
        "matched_positive_chunk_ids": matched_positive_chunk_ids,
        "first_hit_rank": first_hit_rank,
    }

def evaluate_retriever(
    retriever_name: str,
    retrieve_fn: RetrieveFn,
    eval_items: list[dict[str, Any]],
    top_k: int = 5,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """
    评估某一个检索器。

    retriever_name:
        dense 或 sparse。

    retrieve_fn:
        dense_retrieve_by_query 或 sparse_retrieve_by_query。
    """
    metric_names = (
        [f"hit@{k}" for k in HIT_KS]
        + [f"recall@{k}" for k in RECALL_KS]
        + [f"mrr@{MRR_K}"]
    )

    totals = {metric_name: 0.0 for metric_name in metric_names}
    details = []
    error_count = 0

    for case_idx, item in enumerate(eval_items, start=1):
        query = item["query"]
        positive_chunk_ids = item["positive_chunk_ids"]

        print(
            f"[{retriever_name}] {case_idx}/{len(eval_items)} query={query[:40]}",
            flush=True,
        )

        try:
            result = retrieve_fn(
                query=query,
                doc_id=doc_id,
                top_k=top_k,
            )

            hits = result.get("hits", [])
            print(
                f"[{retriever_name}] {case_idx}/{len(eval_items)} done, hits={len(hits)}",
                flush=True,
            )
            metrics = calc_one_query_metrics(
                hits=hits,
                positive_chunk_ids=positive_chunk_ids,
            )

            for metric_name in metric_names:
                totals[metric_name] += metrics[metric_name]

            hit_analysis = analyze_retrieval_hits(
                hits=hits,
                positive_chunk_ids=positive_chunk_ids,
            )

            details.append(
                {
                    "line_no": item["line_no"],
                    "query_id": item["query_id"],
                    "query": query,
                    "query_type": item["query_type"],
                    "note": item.get("note", ""),
                    "positive_chunk_ids": positive_chunk_ids,
                    "positive_doc_ids": item.get("positive_doc_ids", []),
                    "retriever": retriever_name,
                    "metrics": metrics,
                    "matched_positive_chunk_ids": hit_analysis["matched_positive_chunk_ids"],
                    "first_hit_rank": hit_analysis["first_hit_rank"],
                    "top_hits": [
                        {
                            "rank": idx,
                            "chunk_id": hit.get("chunk_id"),
                            "score": hit.get("score"),
                            "source": hit.get("source"),
                            "text_preview": str(hit.get("text", ""))[:120],
                        }
                        for idx, hit in enumerate(hits[:top_k], start=1)
                    ],
                    "error": None,
                }
            )

        except Exception as e:
            error_count += 1

            details.append(
                {
                    "line_no": item["line_no"],
                    "query_id": item["query_id"],
                    "query": query,
                    "query_type": item["query_type"],
                    "note": item.get("note", ""),
                    "positive_chunk_ids": positive_chunk_ids,
                    "positive_doc_ids": item.get("positive_doc_ids", []),
                    "retriever": retriever_name,
                    "metrics": {metric_name: 0.0 for metric_name in metric_names},
                    "matched_positive_chunk_ids": [],
                    "first_hit_rank": None,
                    "top_hits": [],
                    "error": repr(e),
                }
            )
    total_cases = len(eval_items)

    summary = {
        metric_name: round(metric_total / total_cases, 4)
        for metric_name, metric_total in totals.items()
    }

    return {
        "retriever": retriever_name,
        "total_cases": total_cases,
        "error_count": error_count,
        "summary": summary,
        "details": details,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    """打印 dense / sparse 的评估汇总表。"""
    print("\n=== Retrieval Evaluation Summary ===")
    print(
        f"{'retriever':<10} "
        f"{'Hit@1':>8} "
        f"{'Hit@3':>8} "
        f"{'Hit@5':>8} "
        f"{'Recall@3':>10} "
        f"{'Recall@5':>10} "
        f"{'MRR@5':>8} "
        f"{'errors':>8}"
    )
    print("-" * 82)

    for result in results:
        summary = result["summary"]

        print(
            f"{result['retriever']:<10} "
            f"{summary['hit@1']:>8.4f} "
            f"{summary['hit@3']:>8.4f} "
            f"{summary['hit@5']:>8.4f} "
            f"{summary['recall@3']:>10.4f} "
            f"{summary['recall@5']:>10.4f} "
            f"{summary['mrr@5']:>8.4f} "
            f"{result['error_count']:>8}"
        )



def save_results(results: list[dict[str, Any]], output_file: Path) -> None:
    """保存每条 query 的评估明细。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and sparse retrieval.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--doc-id", type=str, default=None)

    args = parser.parse_args()

    if args.top_k < MRR_K:
        raise ValueError(f"top_k 至少要 >= {MRR_K}，否则无法计算 MRR@{MRR_K}")

    eval_items = load_eval_set(args.eval_file)

    results = [
        evaluate_retriever(
            retriever_name="dense",
            retrieve_fn=dense_retrieve_by_query,
            eval_items=eval_items,
            top_k=args.top_k,
            doc_id=args.doc_id,
        ),
        evaluate_retriever(
            retriever_name="sparse",
            retrieve_fn=sparse_retrieve_by_query,
            eval_items=eval_items,
            top_k=args.top_k,
            doc_id=args.doc_id,
        ),
        evaluate_retriever(
            retriever_name="hybrid",
            retrieve_fn=hybrid_retrieve_by_query,
            eval_items=eval_items,
            top_k=args.top_k,
            doc_id=args.doc_id,
        ),
        evaluate_retriever(
            retriever_name="hybrid_rerank",
            retrieve_fn=hybrid_rerank_retrieve_for_eval,
            eval_items=eval_items,
            top_k=args.top_k,
            doc_id=args.doc_id,
        ),
    ]
    print_summary(results)
    save_results(results, args.output_file)

    print(f"\n评估明细已保存: {args.output_file}")


if __name__ == "__main__":
    main()
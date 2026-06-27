import json
import sys
from pathlib import Path
from collections import Counter


def get_project_root() -> Path:
    """
    通过当前脚本位置定位项目根目录，避免 PyCharm 工作目录不一致导致路径错误。
    """
    return Path(__file__).resolve().parents[1]


def load_json_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_chunks(chunks_dir: Path) -> dict:
    """
    扫描 data/processed/chunks/*.json，收集所有 chunk_id。

    返回：
    {
        chunk_id: {
            "doc_id": ...,
            "chunk_index": ...,
            "file_name": ...,
            "source": ...,
            "text": ...,
            "chunk_file": ...
        }
    }
    """
    chunk_id_map = {}

    if not chunks_dir.exists():
        raise FileNotFoundError(f"chunks 目录不存在：{chunks_dir}")

    chunk_files = list(chunks_dir.glob("*.json"))

    if not chunk_files:
        raise FileNotFoundError(f"chunks 目录下没有 json 文件：{chunks_dir}")

    duplicate_chunk_ids = []

    for chunk_file in chunk_files:
        data = load_json_file(chunk_file)
        chunks = data.get("chunks", [])

        if not isinstance(chunks, list):
            print(f"[WARN] 跳过异常 chunk 文件，chunks 不是 list：{chunk_file}")
            continue

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                print(f"[WARN] 发现没有 chunk_id 的 chunk：{chunk_file}")
                continue

            if chunk_id in chunk_id_map:
                duplicate_chunk_ids.append(chunk_id)

            chunk_id_map[chunk_id] = {
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index"),
                "file_name": chunk.get("file_name"),
                "source": chunk.get("source"),
                "text": chunk.get("text", ""),
                "chunk_file": str(chunk_file),
            }

    if duplicate_chunk_ids:
        print("[WARN] 发现重复 chunk_id：")
        for chunk_id in duplicate_chunk_ids[:10]:
            print(f"  - {chunk_id}")

    return chunk_id_map


def load_eval_set(eval_path: Path) -> list:
    """
    读取 retrieval_eval_set.jsonl。
    一行一个 JSON。
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"评估集文件不存在：{eval_path}")

    rows = []

    with eval_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{e}")

            item["_line_no"] = line_no
            rows.append(item)

    if not rows:
        raise ValueError(f"评估集为空：{eval_path}")

    return rows


def validate_eval_set(eval_rows: list, chunk_id_map: dict) -> dict:
    errors = []
    warnings = []

    required_fields = [
        "query_id",
        "query",
        "query_type",
        "positive_chunk_ids",
    ]

    query_ids = []
    query_type_counter = Counter()

    for row in eval_rows:
        line_no = row.get("_line_no")
        query_id = row.get("query_id")

        for field in required_fields:
            if field not in row:
                errors.append(f"第 {line_no} 行缺少字段：{field}")

        if query_id:
            query_ids.append(query_id)

        query_type = row.get("query_type")
        if query_type:
            query_type_counter[query_type] += 1

        positive_chunk_ids = row.get("positive_chunk_ids")

        if not isinstance(positive_chunk_ids, list):
            errors.append(f"第 {line_no} 行 positive_chunk_ids 必须是 list")
            continue

        if len(positive_chunk_ids) == 0:
            errors.append(f"第 {line_no} 行 positive_chunk_ids 为空")

        for chunk_id in positive_chunk_ids:
            if chunk_id not in chunk_id_map:
                errors.append(
                    f"第 {line_no} 行 query_id={query_id} 引用了不存在的 chunk_id：{chunk_id}"
                )

    duplicate_query_ids = [
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    ]

    if duplicate_query_ids:
        for query_id in duplicate_query_ids:
            errors.append(f"query_id 重复：{query_id}")

    return {
        "errors": errors,
        "warnings": warnings,
        "query_type_counter": query_type_counter,
    }


def preview_positive_chunks(eval_rows: list, chunk_id_map: dict, max_rows: int = 3):
    """
    打印前几条样本对应的 positive chunk，方便人工确认。
    """
    print("\n========== Positive Chunk 预览 ==========")

    for row in eval_rows[:max_rows]:
        print(f"\nquery_id: {row.get('query_id')}")
        print(f"query: {row.get('query')}")
        print("positive chunks:")

        for chunk_id in row.get("positive_chunk_ids", []):
            chunk = chunk_id_map.get(chunk_id)

            if not chunk:
                print(f"  - {chunk_id} [不存在]")
                continue

            text = chunk.get("text", "").replace("\n", " ")
            text_preview = text[:120] + ("..." if len(text) > 120 else "")

            print(
                f"  - {chunk_id} | "
                f"doc_id={chunk.get('doc_id')} | "
                f"chunk_index={chunk.get('chunk_index')} | "
                f"source={chunk.get('source')}"
            )
            print(f"    text: {text_preview}")


def main():
    project_root = get_project_root()

    eval_path = project_root / "data" / "eval" / "retrieval_eval_set.jsonl"
    chunks_dir = project_root / "data" / "processed" / "chunks"

    print("========== Retrieval Eval Set Check ==========")
    print(f"project_root: {project_root}")
    print(f"eval_path: {eval_path}")
    print(f"chunks_dir: {chunks_dir}")

    try:
        chunk_id_map = load_all_chunks(chunks_dir)
        eval_rows = load_eval_set(eval_path)
        result = validate_eval_set(eval_rows, chunk_id_map)

        print("\n========== 基本统计 ==========")
        print(f"chunk 总数: {len(chunk_id_map)}")
        print(f"eval query 总数: {len(eval_rows)}")
        print("query_type 分布:")

        for query_type, count in result["query_type_counter"].items():
            print(f"  - {query_type}: {count}")

        if result["warnings"]:
            print("\n========== WARNINGS ==========")
            for warning in result["warnings"]:
                print(f"[WARN] {warning}")

        if result["errors"]:
            print("\n========== ERRORS ==========")
            for error in result["errors"]:
                print(f"[ERROR] {error}")

            print("\n校验失败：请先修复评估集或 chunks 数据。")
            sys.exit(1)

        preview_positive_chunks(eval_rows, chunk_id_map)

        print("\n========== CHECK PASSED ==========")
        print("评估集格式正确，positive_chunk_ids 均能在本地 chunks 中找到。")

    except Exception as e:
        print("\n========== CHECK FAILED ==========")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
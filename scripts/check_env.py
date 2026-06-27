from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.config.settings import settings


def main():
    print("========== Project Env Check ==========")
    print(f"Project name: {settings.project_name}")
    print(f"Env: {settings.env}")

    print("\n[LLM]")
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model}")
    print(f"Timeout: {settings.llm_timeout}")

    print("\n[Milvus]")
    print(f"URI: {settings.milvus_url}")
    print(f"DB name: {settings.milvus_db_name}")
    print(f"Collection: {settings.milvus_collection_name}")
    print(f"Vector dim: {settings.milvus_vector_dim}")
    print(f"Metric type: {settings.milvus_metric_type}")

    print("\n[Embedding]")
    print(f"Embedding path: {settings.embedding_model_path}")
    print(f"Embedding device: {settings.embedding_device}")

    print("\n[Reranker]")
    print(f"Reranker path: {settings.reranker_model_path}")
    print(f"Reranker device: {settings.reranker_device}")

    print("\n[Paths]")
    print(f"Data dir: {settings.data_dir}")
    print(f"Processed data dir: {settings.processed_data_dir}")
    print(f"Upload dir: {settings.upload_dir}")
    print(f"Log dir: {settings.log_dir}")

    print("\nCheck env: OK")
    print("=======================================")


if __name__ == "__main__":
    main()
import os
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from app.config.settings import settings


def check_path(name, path):
    if not path:
        raise ValueError(f"{name} path is empty")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} path not found: {path}")

    print(f"{name} path OK: {path}")


def main():
    print("========== Local Models Check ==========")

    check_path("Embedding model", settings.embedding_model_path)
    check_path("Reranker model", settings.reranker_model_path)

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(
        settings.embedding_model_path,
        device=settings.embedding_device,
    )

    text = "SmartEnv-Gateway SE-100 技术支持系统用于智能硬件故障诊断和工单处理。"
    vector = embed_model.encode(text)

    print(f"Embedding vector dim: {len(vector)}")

    if len(vector) != settings.milvus_vector_dim:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vector)}, "
            f"expected {settings.milvus_vector_dim}"
        )

    print("\nLoading reranker model...")
    reranker = FlagReranker(
        settings.reranker_model_path,
        use_fp16=False,
    )

    query = "什么是个人知识库助手？"
    passage = "Personal Knowledge Assistant 是一个个人知识库助手项目。"

    score = reranker.compute_score([[query, passage]])

    print(f"Reranker score: {score}")
    print("Local models check: OK")
    print("========================================")


if __name__ == "__main__":
    main()
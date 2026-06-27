import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, __version__ as pymilvus_version


def main():
    load_dotenv()

    uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    timeout = int(os.getenv("MILVUS_TIMEOUT", "10"))

    print("========== Milvus Self Check ==========")
    print(f"PyMilvus version: {pymilvus_version}")
    print(f"Milvus URI: {uri}")

    client = MilvusClient(
        uri=uri,
        timeout=timeout,
    )

    collections = client.list_collections()

    print("Connection: OK")
    print(f"Collections count: {len(collections)}")

    if collections:
        print("Collections:")
        for name in collections:
            print(f"  - {name}")
    else:
        print("Collections: empty")

    print("=======================================")


if __name__ == "__main__":
    main()
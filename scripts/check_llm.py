from openai import OpenAI
from app.config.settings import settings


def main():
    print("========== LLM Self Check ==========")
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model}")

    client = OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "请只回复：LLM连接正常"},
        ],
        temperature=0,
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("====================================")


if __name__ == "__main__":
    main()
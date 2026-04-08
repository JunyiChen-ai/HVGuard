from openai import OpenAI
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

def main():
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")

    print("BASE_URL =", base_url)
    print("MODEL =", model)
    print("API_KEY_SET =", bool(api_key))
    if api_key:
        print("API_KEY_PREFIX =", api_key[:12])

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly: ok"}
            ],
            max_tokens=8,
        )
        print("SUCCESS")
        print(resp.choices[0].message.content)
    except Exception as e:
        print("ERROR")
        print(type(e).__name__)
        print(e)
        print("TRACEBACK_START")
        traceback.print_exc()
        print("TRACEBACK_END")


if __name__ == "__main__":
    main()

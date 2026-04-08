import base64
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


def main():
    image_path = "framework_img.png"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "qwen2.5-vl-72b-instruct"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please briefly describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        print("SUCCESS")
        print(resp.choices[0].message.content)
    except Exception as e:
        print("ERROR")
        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()

import argparse
import asyncio
import base64
import json
import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


async def analyze_video_with_gpt4(prompt):
    instruction = "This is a video, which may contain harmful content such as hate speech, explicit violence, discrimination, or other forms of harmful behavior. You are a content moderation specialist. Your task is to identify any instances of hate speech, violent imagery, discriminatory actions, or any other content that could be considered harmful, abusive, or offensive. "

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            print(f"Rate limit exceeded, retrying {attempt+1}/{max_retries}...")
        except APIConnectionError:
            print(f"Connection error, retrying {attempt+1}/{max_retries}...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return "Error: Unable to analyze video"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def describe_video_frames(frames, prompt):
    content = [{"type": "text", "text": prompt}]
    for frame_path in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame_path)}"
                },
            }
        )

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing frames: {e}")
        return ""


async def describe_video_text(prompt):
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Text description error: {e}")
        return ""


def load_data(data_path, save_path):
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as file:
            saved_data = json.load(file)
        saved_map = {
            item.get("Video_ID"): item for item in saved_data if item.get("Video_ID")
        }
        for item in data:
            saved_item = saved_map.get(item.get("Video_ID"))
            if saved_item:
                item.update(saved_item)
    return data


async def process_item(item, data, save_path, write_lock, semaphore, quad_root="./datasets/HateMM/quad"):
    video_id = item.get("Video_ID")
    video_title = item.get("Title")
    video_transcript = item.get("Transcript")
    video_emotion = item.get("Emotion")
    video_path = os.path.join(quad_root, video_id)

    if item.get("Mix_description"):
        print(f"Skipping video: {video_id} (already processed)")
        return

    if os.path.isdir(video_path):
        frames_list = [
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".png"))
        ]
        if not frames_list:
            print(f"No image files found in {video_path}, skipping this video.")
            return
        video_frames = sorted(frames_list)
    else:
        print(f"Directory {video_path} does not exist, skipping this video.")
        return

    print(f"Processing video: {video_id} ...")
    video_start_time = time.time()

    async with semaphore:
        vision_prompt = (
            "You are analyzing a video represented by multiple 2x2 quad images. "
            "Each quad contains four consecutive frames arranged in temporal order: "
            "top-left, top-right, bottom-left, bottom-right. "
            "The quads themselves are also provided in chronological order and together "
            "represent the full video. Describe the visual content of the video concisely, "
            "paying attention to any special characters or text, and mention the visible "
            "people, objects, and notable temporal changes."
        )
        item["Frames_description"] = await describe_video_frames(video_frames, vision_prompt)

        if not video_transcript or video_transcript == "None":
            item["Text_description"] = "None"
        else:
            text_prompt = f"""The title of the video is "{video_title}". The transcript in the video is as follows: "{video_transcript}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise."""
            item["Text_description"] = await describe_video_text(text_prompt)

        video_vision = item.get("Frames_description")
        video_text = item.get("Text_description")

        mix_prompt = f"""
        Please combine the video title, trainscript, voice emotion, video text description, video frames description and analyze both the visual, textual and audio elements of the video to detect and flag any hateful content.
        No need to describe the content of video, only answer implicit meanings and whether this video expresses hateful content further. Ensure the accuracy of the answer and try to be concise as much as possible.
        Video title:"{video_title}"
        Trainscript:"{video_transcript}"
        Voice emotion:"{video_emotion}" 
        Video text description:"{video_text}"
        Video frames description:"{video_vision}" 
        Output format:
        [implicit meanings]:<Your analysis of implicit meanings>
        [Hate or not]:<Hateful, Offensive or Normal, just one word>
        """
        item["Mix_description"] = await analyze_video_with_gpt4(mix_prompt)

    print("Done!")
    video_end_time = time.time()
    print(f"Time taken for video {video_id}: {video_end_time - video_start_time:.2f} seconds")

    async with write_lock:
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


async def process(data_path, save_path, max_concurrent, quad_root="./datasets/HateMM/quad"):
    data = load_data(data_path, save_path)
    total_start_time = time.time()

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    progress_bar = tqdm(total=len(data), desc="Processing videos", unit="video")

    async def wrapped_process_item(item):
        try:
            await process_item(item, data, save_path, write_lock, semaphore, quad_root=quad_root)
        finally:
            progress_bar.update(1)

    tasks = [
        wrapped_process_item(item)
        for item in data
    ]
    await asyncio.gather(*tasks)
    progress_bar.close()

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    total_end_time = time.time()
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generating video descriptions using MLLM...")
    parser.add_argument("--dataset_name", type=str, default="Multihateclip",
                        choices=["Multihateclip", "HateMM"], help="Dataset name")
    parser.add_argument("--language", type=str, default="English",
                        choices=["Chinese", "English"], help="Language of the dataset")
    parser.add_argument("--max_concurrent", type=int, default=4,
                        help="Maximum number of concurrent video processing tasks")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    language = args.language

    if dataset_name == "Multihateclip":
        data_path = f"./datasets/Multihateclip/{language}/annotation(new).json"
        save_path = f"./datasets/Multihateclip/{language}/data.json"
        quad_root = f"./datasets/Multihateclip/{language}/quad"
    else:
        data_path = "./datasets/HateMM/annotation(re).json"
        save_path = "./datasets/HateMM/data.json"
        quad_root = "./datasets/HateMM/quad"

    asyncio.run(process(data_path, save_path, args.max_concurrent, quad_root=quad_root))
    print(f"Results have been saved to {save_path}.")


if __name__ == "__main__":
    main()

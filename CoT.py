import torch
import argparse
import os
import csv
from PIL import Image
import json
from openai import OpenAI, APIConnectionError, RateLimitError
import time

import base64
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

def analyze_video_with_gpt4(prompt):
    """
    Use the GPT-4 API to perform content analysis on a video.

    Args:
        title (str): Video title
        transcript (str): Concatenated title and transcript of the video
        frames_description (str): Description of the video frames
    """


    instruction = "This is a video, which may contain harmful content such as hate speech, explicit violence, discrimination, or other forms of harmful behavior. You are a content moderation specialist. Your task is to identify any instances of hate speech, violent imagery, discriminatory actions, or any other content that could be considered harmful, abusive, or offensive. "

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            print(
                f"Rate limit exceeded, retrying {attempt+1}/{max_retries}...")
        except APIConnectionError:
            print(f"Connection error, retrying {attempt+1}/{max_retries}...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return "Error: Unable to analyze video"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_video_frames(frames, prompt):
    descriptions = []
    for frame_path in frames:
        base64_image = encode_image(frame_path)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            description = response.choices[0].message.content.strip()
            descriptions.append(description)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            descriptions.append("")
    return descriptions


def describe_video_text(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        description = response.choices[0].message.content.strip()
        return [description]
    except Exception as e:
        print(f"Text description error: {e}")
        return [""]


def process(data_path, save_path, frame_step):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            saved_data = json.load(file)
        saved_map = {
            item.get("Video_ID"): item for item in saved_data if item.get("Video_ID")
        }
        for item in data:
            saved_item = saved_map.get(item.get("Video_ID"))
            if saved_item:
                item.update(saved_item)

    total_start_time = time.time()

    for item in data:
        video_id = item.get("Video_ID")
        video_title = item.get("Title")
        video_transcript = item.get("Transcript")
        video_emotion = item.get("Emotion")
        video_path = os.path.join(
            f'./datasets/HateMM/frames', video_id)
        video_vision = item.get("Frames_description")
        video_text = item.get("Text_description")
        video_label = item.get("Label")

        if item.get("Mix_description"):
            print(f"Skipping video: {video_id} (already processed)")
            continue

        if os.path.isdir(video_path):
            frames_list = [
                os.path.join(video_path, f) for f in os.listdir(video_path)
                if f.lower().endswith(('.jpg', '.png'))
            ]
            if not frames_list:
                print(
                    f"No image files found in {video_path}, skipping this video.")
                continue
            video_frames = sorted(frames_list)
        else:
            print(
                f"Directory {video_path} does not exist, skipping this video.")
            continue

        print(f"Processing video: {video_id} ...")
        video_start_time = time.time()

        vision_prompt = f"""Describe the content of the video concisely, paying attention to any special characters or text, be sure to mention their name."""
        frames_output = describe_video_frames(video_frames, vision_prompt)
        item["Frames_description"] = frames_output[0]

        if not video_transcript or video_transcript == "None":
            item["Text_description"] = "None"
        else:
            text_prompt = f"""The title of the video is "{video_title}". The transcript in the video is as follows: "{video_transcript}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise."""
            text_output = describe_video_text(text_prompt)
            item["Text_description"] = text_output[0]

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
        mix_output = analyze_video_with_gpt4(mix_prompt)
        item["Mix_description"] = mix_output

        print("Done!")

        video_end_time = time.time()
        print(
            f"Time taken for video {video_id}: {video_end_time - video_start_time:.2f} seconds")

        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    total_end_time = time.time()
    print(
        f"Total processing time: {total_end_time - total_start_time:.2f} seconds")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generating video descriptions using MLLM...")
    parser.add_argument('--dataset_name', type=str, default='Multihateclip',
                        choices=['Multihateclip', 'HateMM'], help='Dataset name')
    parser.add_argument('--language', type=str, default='English',
                        choices=['Chinese', 'English'], help='Language of the dataset')

    return parser.parse_args()

def main():
    args = parse_args()
    Dataset_name = [args.dataset_name]  # (Multihateclip, HateMM)
    Language = [args.language]  # (Chinese, English)
    if Dataset_name[0] == 'Multihateclip':
        data_path = f'./datasets/Multihateclip/'+Language[0]+f'/annotation(new).json'
        save_path = f'./datasets/Multihateclip/'+Language[0]+f'/data.json'
    elif Dataset_name[0] == 'HateMM':
        data_path = f'./datasets/HateMM/annotation(re).json'
        save_path = f'./datasets/HateMM/data.json'
        
    frame_step = 1

    process(data_path, save_path, frame_step)
    print(f"Results have been saved to {save_path}.")


if __name__ == "__main__":
    main()


"""
Preprocess Multihateclip datasets: extract frames, audio, update annotation.
Handles three cases per video:
  1. Folder with video.mp4/video.webm inside → extract frames + audio
  2. Folder with only frame PNGs (no video) → use existing frames, skip audio
  3. Empty folder or non-existent → skip entirely (will be filtered)
"""
import argparse
import cv2
import json
import os
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


def slice_frames(video_path, output_dir, num_frames=32):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_frames:
        return True
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    if num_frames <= total_frames:
        seg_size = (total_frames - 1) / num_frames
        selected_ids = [int(np.round(seg_size * i)) for i in range(num_frames)]
    else:
        selected_ids = list(range(total_frames)) * (num_frames // total_frames + 1)
        selected_ids = selected_ids[:num_frames]

    count = 0
    saved_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if count in selected_ids:
            image_name = f"frame_{saved_count + 1:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, image_name), frame)
            saved_count += 1
        count += 1

    cap.release()
    return saved_count > 0


def extract_audio_ffmpeg(video_path, audio_path):
    if os.path.exists(audio_path):
        return True
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, timeout=60
        )
        return result.returncode == 0 and os.path.exists(audio_path)
    except Exception:
        return False


def process_dataset(lang, video_dir_name):
    dataset_root = Path(f"datasets/Multihateclip/{lang}")
    annotation_path = dataset_root / "annotation(new).json"
    video_base = Path(f"/home/junyi/HVideo/{video_dir_name}")
    frames_root = dataset_root / "frames"
    audios_root = dataset_root / "audios"
    frames_root.mkdir(parents=True, exist_ok=True)
    audios_root.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"video_extracted": 0, "frames_existing": 0, "skipped": 0, "audio_ok": 0}

    for item in tqdm(data, desc=f"Preprocessing {lang}"):
        vid = item["Video_ID"]
        folder = video_base / vid
        frames_out = frames_root / vid
        audio_out = audios_root / f"{vid}.wav"

        # Find video file
        video_file = None
        if folder.is_dir():
            for vname in ["video.mp4", "video.webm"]:
                candidate = folder / vname
                if candidate.exists():
                    video_file = candidate
                    break

        if video_file:
            # Case 1: has video file → extract frames + audio
            if slice_frames(video_file, str(frames_out)):
                item["Frames_path"] = str(frames_out)
                stats["video_extracted"] += 1
            else:
                item["Frames_path"] = ""

            if extract_audio_ffmpeg(video_file, str(audio_out)):
                item["Audio_path"] = str(audio_out)
                stats["audio_ok"] += 1
            else:
                item["Audio_path"] = ""

        elif folder.is_dir():
            # Case 2: folder exists, check for existing frames (PNGs from missing zip)
            pngs = sorted([f for f in folder.iterdir()
                          if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            if pngs:
                # Symlink to use existing frames directly
                if not frames_out.exists():
                    frames_out.symlink_to(folder)
                item["Frames_path"] = str(frames_out)
                item["Audio_path"] = ""  # No audio available
                stats["frames_existing"] += 1
            else:
                item["Frames_path"] = ""
                item["Audio_path"] = ""
                stats["skipped"] += 1
        else:
            item["Frames_path"] = ""
            item["Audio_path"] = ""
            stats["skipped"] += 1

    # Save updated annotation
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{lang}: {stats}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["Chinese", "English", "both"], default="both")
    args = parser.parse_args()

    os.chdir("/home/junyi/HVGuard")

    if args.lang in ("Chinese", "both"):
        process_dataset("Chinese", "Bilibili")
    if args.lang in ("English", "both"):
        process_dataset("English", "YouTube")


if __name__ == "__main__":
    main()

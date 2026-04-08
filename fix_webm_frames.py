"""Fix frame extraction for webm videos that cv2 failed on. Uses ffmpeg."""
import json
import os
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_frames_ffmpeg(video_path, output_dir, num_frames=32):
    os.makedirs(output_dir, exist_ok=True)
    # Get total frames using ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        total_frames = int(result.stdout.strip())
    except Exception:
        # Fallback: just extract uniformly by time
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration,r_frame_rate", "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            lines = result.stdout.strip().split('\n')
            parts = lines[0].split(',')
            fps_str = parts[0]
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            duration = float(parts[1]) if len(parts) > 1 else 10.0
            total_frames = int(fps * duration)
        except Exception:
            total_frames = 320  # fallback

    if total_frames <= 0:
        total_frames = 320

    if num_frames <= total_frames:
        seg_size = (total_frames - 1) / num_frames
        selected_ids = [int(np.round(seg_size * i)) for i in range(num_frames)]
    else:
        selected_ids = list(range(total_frames)) * (num_frames // total_frames + 1)
        selected_ids = selected_ids[:num_frames]

    # Use ffmpeg select filter
    select_expr = "+".join([f"eq(n\\,{idx})" for idx in selected_ids])
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"select='{select_expr}',setpts=N/TB",
             "-vsync", "vfr", "-q:v", "2",
             os.path.join(output_dir, "frame_%03d.jpg")],
            capture_output=True, timeout=120
        )
    except Exception:
        pass

    return len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])


def main():
    os.chdir("/home/junyi/HVGuard")
    video_base = Path("/home/junyi/HVideo/YouTube")

    with open("datasets/Multihateclip/English/annotation(new).json") as f:
        data = json.load(f)

    fixed = 0
    for item in tqdm(data, desc="Fixing webm frames"):
        vid = item["Video_ID"]
        frames_dir = f"datasets/Multihateclip/English/frames/{vid}"

        # Skip if already has frames
        if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
            continue

        # Find video file
        folder = video_base / vid
        video_file = None
        for vname in ["video.mp4", "video.webm"]:
            candidate = folder / vname
            if candidate.exists():
                video_file = candidate
                break

        if not video_file:
            continue

        count = extract_frames_ffmpeg(video_file, frames_dir)
        if count > 0:
            item["Frames_path"] = frames_dir
            fixed += 1

    with open("datasets/Multihateclip/English/annotation(new).json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed {fixed} videos")


if __name__ == "__main__":
    main()

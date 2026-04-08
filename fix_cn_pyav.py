import av
import json
import os
import numpy as np
from pathlib import Path

def extract_frames_pyav(video_path, output_dir, num_frames=32):
    os.makedirs(output_dir, exist_ok=True)
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames <= 0:
            total_frames = 0
            for _ in container.decode(video=0):
                total_frames += 1
            container.close()
            container = av.open(str(video_path))
        if total_frames <= 0:
            container.close()
            return 0
        if num_frames <= total_frames:
            seg_size = (total_frames - 1) / num_frames
            selected_ids = set(int(np.round(seg_size * i)) for i in range(num_frames))
        else:
            selected_ids = set(range(total_frames))
        saved = 0
        for i, frame in enumerate(container.decode(video=0)):
            if i in selected_ids:
                img = frame.to_image()
                img.save(os.path.join(output_dir, f"frame_{saved+1:03d}.jpg"))
                saved += 1
                if saved >= num_frames:
                    break
        container.close()
        return saved
    except Exception as e:
        print(f"Error on {video_path}: {e}", flush=True)
        return 0

def main():
    os.chdir("/home/junyi/HVGuard")
    video_base = Path("/home/junyi/HVideo/Bilibili")
    with open("datasets/Multihateclip/Chinese/annotation(new).json") as f:
        data = json.load(f)
    todo = []
    for item in data:
        vid = item["Video_ID"]
        frames_dir = f"datasets/Multihateclip/Chinese/frames/{vid}"
        jpg_count = len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg','.png'))]) if os.path.exists(frames_dir) else 0
        if jpg_count > 0:
            continue
        folder = video_base / vid
        for vname in ["video.mp4", "video.webm"]:
            c = folder / vname
            if c.exists():
                todo.append((item, str(c), frames_dir))
                break
    print(f"Need to fix: {len(todo)} videos", flush=True)
    fixed = 0
    for idx, (item, vfile, frames_dir) in enumerate(todo):
        count = extract_frames_pyav(vfile, frames_dir)
        if count > 0:
            item["Frames_path"] = frames_dir
            fixed += 1
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(todo)}, fixed: {fixed}", flush=True)
    with open("datasets/Multihateclip/Chinese/annotation(new).json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Done. Fixed {fixed}/{len(todo)} Chinese videos", flush=True)

if __name__ == "__main__":
    main()

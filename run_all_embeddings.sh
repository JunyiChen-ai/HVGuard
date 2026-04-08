#!/bin/bash
set -e
cd /home/junyi/HVGuard

echo "=== [1/8] Chinese text_features ==="
python embeddings/text_embedding.py \
  --json datasets/Multihateclip/Chinese/data.json \
  --out embeddings/Multihateclip/Chinese/text_features.pth \
  --field text 2>&1 | tail -3

echo "=== [2/8] Chinese MLLM_rationale_features ==="
python embeddings/text_embedding.py \
  --json datasets/Multihateclip/Chinese/data.json \
  --out embeddings/Multihateclip/Chinese/MLLM_rationale_features.pth \
  --field mix 2>&1 | tail -3

echo "=== [3/8] English text_features ==="
python embeddings/text_embedding.py \
  --json datasets/Multihateclip/English/data.json \
  --out embeddings/Multihateclip/English/text_features.pth \
  --field text 2>&1 | tail -3

echo "=== [4/8] English MLLM_rationale_features ==="
python embeddings/text_embedding.py \
  --json datasets/Multihateclip/English/data.json \
  --out embeddings/Multihateclip/English/MLLM_rationale_features.pth \
  --field mix 2>&1 | tail -3

echo "=== [5/8] Chinese audio_features ==="
python embeddings/audio_embedding.py \
  --json datasets/Multihateclip/Chinese/data.json \
  --out embeddings/Multihateclip/Chinese/audio_features.pth 2>&1 | tail -3

echo "=== [6/8] English audio_features ==="
python embeddings/audio_embedding.py \
  --json datasets/Multihateclip/English/data.json \
  --out embeddings/Multihateclip/English/audio_features.pth 2>&1 | tail -3

echo "=== [7/8] Chinese frame_features ==="
python embeddings/frames_embedding.py \
  --json datasets/Multihateclip/Chinese/data.json \
  --out embeddings/Multihateclip/Chinese/frame_features.pth 2>&1 | tail -3

echo "=== [8/8] English frame_features ==="
python embeddings/frames_embedding.py \
  --json datasets/Multihateclip/English/data.json \
  --out embeddings/Multihateclip/English/frame_features.pth 2>&1 | tail -3

echo "=== ALL DONE ==="

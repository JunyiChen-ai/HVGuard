#!/bin/bash
set -e
cd /home/junyi/HVGuard

echo "=========================================="
echo "=== Chinese 2-class ==="
echo "=========================================="
python HVGuard.py --dataset_name Multihateclip --language Chinese --num_classes 2 --mode train --split_mode fixed

echo "=========================================="
echo "=== Chinese 3-class ==="
echo "=========================================="
python HVGuard.py --dataset_name Multihateclip --language Chinese --num_classes 3 --mode train --split_mode fixed

echo "=========================================="
echo "=== English 2-class ==="
echo "=========================================="
python HVGuard.py --dataset_name Multihateclip --language English --num_classes 2 --mode train --split_mode fixed

echo "=========================================="
echo "=== English 3-class ==="
echo "=========================================="
python HVGuard.py --dataset_name Multihateclip --language English --num_classes 3 --mode train --split_mode fixed

echo "=========================================="
echo "=== ALL TRAINING DONE ==="
echo "=========================================="

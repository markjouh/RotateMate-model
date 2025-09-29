#!/bin/bash
set -e

echo "Setting up environment..."

# System-wide installation (no venv)
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt

mkdir -p data/{raw,coco,shards} checkpoints logs exports

echo "Setup complete. Run: python3 run.py --steps all"
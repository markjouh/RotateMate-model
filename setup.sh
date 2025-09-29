#!/bin/bash
set -e

echo "Setting up environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

mkdir -p data/{raw,coco,shards} checkpoints logs exports

echo "Setup complete. Run: source venv/bin/activate && python run.py --steps all"
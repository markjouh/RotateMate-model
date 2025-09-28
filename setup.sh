#!/bin/bash
set -e

echo "Setting up Python environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/{raw,coco,shards}
mkdir -p checkpoints logs exports

echo "Setup complete. Activate with: source venv/bin/activate"
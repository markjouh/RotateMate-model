#!/bin/bash
set -e


if [ "$(id -u)" -eq 0 ]; then
  echo "Error: Do not run setup.sh with sudo or as root. Run as the target user so directories remain writable." >&2
  exit 1
fi

echo "Setting up environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

mkdir -p data/{raw,coco} checkpoints logs exports

for dir in data checkpoints logs exports; do
  if [ ! -w "$dir" ]; then
    echo "Directory $dir is not writable by $(whoami). Fix with: chmod -R u+rwX $dir" >&2
    exit 1
  fi
done

echo "Setup complete. Run: source venv/bin/activate && python run.py --steps all"

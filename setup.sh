#!/bin/bash
set -e

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download COCO datasets
mkdir -p data
cd data
wget -nc http://images.cocodataset.org/zips/train2017.zip
wget -nc http://images.cocodataset.org/zips/val2017.zip
unzip -n train2017.zip
unzip -n val2017.zip
cd ..

echo "Setup complete! Activate venv with: source venv/bin/activate"
echo "Then run: python train.py"
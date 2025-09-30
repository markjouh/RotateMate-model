#!/bin/bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Download COCO datasets
mkdir -p data
cd data
wget -nc http://images.cocodataset.org/zips/train2017.zip
wget -nc http://images.cocodataset.org/zips/val2017.zip
unzip -n train2017.zip
unzip -n val2017.zip
cd ..

echo "Setup complete! Run: python train.py"
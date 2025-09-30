#!/bin/bash
set -e

# Download COCO datasets
mkdir -p data
cd data
wget -nc http://images.cocodataset.org/zips/train2017.zip
wget -nc http://images.cocodataset.org/zips/val2017.zip
unzip -q -n train2017.zip
unzip -q -n val2017.zip
cd ..

echo "Dataset download complete!"
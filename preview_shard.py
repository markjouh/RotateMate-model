#!/usr/bin/env python3

import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def preview_shard(shard_path, num_samples=4):
    data = torch.load(shard_path, map_location='cpu', weights_only=True)

    images = data['images_u8'][:num_samples]
    labels = data['labels'][:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
    if num_samples == 1:
        axes = [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        img_np = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img_np)
        axes[idx].set_title(f'Rotation: {label.item()*90}°')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Shard: {shard_path}")
    print(f"Total samples: {images.shape[0]}")
    print(f"Image shape: {images[0].shape}")
    print(f"Labels: {labels[:num_samples].tolist()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_shard.py <shard_file.pt> [num_samples]")
        sys.exit(1)

    shard_file = Path(sys.argv[1])
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if not shard_file.exists():
        print(f"Error: {shard_file} not found")
        sys.exit(1)

    preview_shard(shard_file, num_samples)
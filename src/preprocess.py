#!/usr/bin/env python3
"""GPU-accelerated dataset preprocessing using Kornia."""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import kornia as K

from .dataset import _gather_images


def preprocess_split_gpu(image_dir, output_dir, rotations, image_size, batch_size=512):
    """Preprocess images using GPU acceleration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all images
    image_paths = _gather_images([image_dir])
    print(f"Preprocessing {len(image_paths)} images with {len(rotations)} rotations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process in batches
    total_saved = 0
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_ids = []

        # Load and resize each image to same size
        for img_path in batch_paths:
            try:
                img = K.io.load_image(str(img_path), K.io.ImageLoadType.RGB32)
                # Resize to target size immediately
                img = K.geometry.transform.resize(
                    img.unsqueeze(0),
                    (image_size, image_size),
                    interpolation='bilinear',
                    antialias=True
                ).squeeze(0)
                batch_images.append(img)
                batch_ids.append(img_path.stem)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue

        if not batch_images:
            continue

        # Stack and move to GPU
        images = torch.stack(batch_images).to(device)

        # Generate all rotations for this batch
        for rotation_deg in rotations:
            if rotation_deg == 0:
                rotated = images
            else:
                angle = torch.tensor([rotation_deg] * len(images), device=device)
                rotated = K.geometry.transform.rotate(images, angle, padding_mode='zeros')

            # Move back to CPU and save each image
            rotated_cpu = rotated.cpu()
            for j, image_id in enumerate(batch_ids):
                output_path = output_dir / f"{image_id}_rot{rotation_deg}.pt"
                torch.save(rotated_cpu[j], output_path)
                total_saved += 1

    print(f"Saved {total_saved} preprocessed tensors to {output_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated dataset preprocessing")
    parser.add_argument('--config', default='configs/h100.yaml', help='Config file')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for GPU processing')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    rotations = data_cfg.get('rotations', [0, 90, 180, 270])
    image_size = data_cfg.get('image_size', 256)

    # Preprocess each split
    for split_name in ['train2017', 'val2017', 'test2017']:
        split_path = Path(data_cfg['extracted_dir']) / split_name
        if not split_path.exists():
            print(f"Skipping {split_name} (not found)")
            continue

        output_dir = Path(data_cfg['extracted_dir']) / f"{split_name}_preprocessed"
        if output_dir.exists() and len(list(output_dir.glob("*.pt"))) > 0:
            print(f"Skipping {split_name} (already preprocessed)")
            continue

        print(f"\nPreprocessing {split_name}...")
        preprocess_split_gpu(split_path, output_dir, rotations, image_size, args.batch_size)

    print("\nPreprocessing complete! Training will auto-detect and use preprocessed data.")


if __name__ == "__main__":
    main()
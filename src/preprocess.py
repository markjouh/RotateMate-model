#!/usr/bin/env python3
"""GPU-accelerated dataset preprocessing using Kornia."""

import argparse
import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import kornia as K

from .dataset import _gather_images


ROTATIONS = [0, 90, 180, 270]


def preprocess_split_gpu(image_dir, output_dir, image_size=256, batch_size=512):
    """Preprocess images using GPU acceleration.

    Loads images, resizes to target size, applies rotations, and saves as tensors.
    All image processing (resize, rotate) is done on GPU for maximum speed.

    Args:
        image_dir: Directory containing raw images
        output_dir: Directory to save preprocessed tensors
        image_size: Target image size (default 256)
        batch_size: Number of images to process at once (default 512)

    Returns:
        Total number of preprocessed tensors saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _gather_images([image_dir])
    total_expected = len(image_paths) * len(ROTATIONS)

    print(f"Preprocessing {len(image_paths)} images with {len(ROTATIONS)} rotations", flush=True)
    print(f"Will save {total_expected} preprocessed tensors", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    total_saved = 0
    failed_count = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_ids = []

        # Load images and resize on GPU
        for img_path in batch_paths:
            try:
                # Load image on CPU (CHW format, values in [0, 1])
                img = K.io.load_image(str(img_path), K.io.ImageLoadType.RGB32)

                if img.dim() != 3 or img.shape[0] != 3:
                    failed_count += 1
                    continue

                # Move to GPU and resize (much faster than CPU resize)
                img = img.to(device)
                img = K.geometry.transform.resize(
                    img.unsqueeze(0),
                    (image_size, image_size),
                    interpolation='bilinear',
                    antialias=True
                ).squeeze(0)

                batch_images.append(img)
                batch_ids.append(img_path.stem)

            except Exception as e:
                print(f"Failed to load {img_path.name}: {e}", flush=True)
                failed_count += 1
                continue

        if not batch_images:
            continue

        # Stack into batch (already on GPU)
        images = torch.stack(batch_images)

        # Sanity check
        assert images.shape == (len(batch_images), 3, image_size, image_size), \
            f"Unexpected batch shape: {images.shape}"

        # Generate all rotations
        for rotation_deg in ROTATIONS:
            if rotation_deg == 0:
                rotated = images
            else:
                angle = torch.full(
                    (len(images),),
                    float(rotation_deg),
                    dtype=torch.float32,
                    device=device
                )
                rotated = K.geometry.transform.rotate(images, angle, padding_mode='zeros')

            # Save each rotated image
            rotated_cpu = rotated.cpu()
            for j, image_id in enumerate(batch_ids):
                output_path = output_dir / f"{image_id}_rot{rotation_deg}.pt"
                torch.save(rotated_cpu[j], output_path)
                total_saved += 1

    print(f"\nSaved {total_saved}/{total_expected} preprocessed tensors to {output_dir}", flush=True)
    if failed_count > 0:
        print(f"Failed to process {failed_count} images", flush=True)

    return total_saved


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated dataset preprocessing")
    parser.add_argument('--config', default='configs/h100.yaml', help='Config file')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for GPU processing')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    image_size = data_cfg.get('image_size', 256)

    for split_name in ['train2017', 'val2017', 'test2017']:
        split_path = Path(data_cfg['extracted_dir']) / split_name
        if not split_path.exists():
            print(f"Skipping {split_name} (not found)", flush=True)
            continue

        output_dir = Path(data_cfg['extracted_dir']) / f"{split_name}_preprocessed"
        if output_dir.exists() and any(output_dir.glob("*.pt")):
            print(f"Skipping {split_name} (already preprocessed)", flush=True)
            continue

        print(f"\nPreprocessing {split_name}...", flush=True)
        preprocess_split_gpu(split_path, output_dir, image_size=image_size, batch_size=args.batch_size)

    print("\nPreprocessing complete!", flush=True)


if __name__ == "__main__":
    main()
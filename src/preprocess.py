#!/usr/bin/env python3
"""GPU-accelerated dataset preprocessing using Kornia."""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import kornia as K

from .dataset import _gather_images


ROTATIONS = [0, 90, 180, 270]


def load_image(img_path):
    """Load a single image. Used for parallel loading."""
    try:
        img = K.io.load_image(str(img_path), K.io.ImageLoadType.RGB32)
        if img.dim() != 3 or img.shape[0] != 3:
            return None, img_path.stem
        return img, img_path.stem
    except Exception:
        return None, img_path.stem


def preprocess_split_gpu(image_dir, output_dir, image_size=256, batch_size=512, num_workers=8):
    """Preprocess images using GPU acceleration with parallel CPU loading.

    Loads images in parallel on CPU, processes on GPU, saves entire batches to disk.

    Args:
        image_dir: Directory containing raw images
        output_dir: Directory to save preprocessed tensors
        image_size: Target image size (default 256)
        batch_size: Number of images to process at once (default 512)
        num_workers: Number of threads for parallel image loading (default 8)

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
    print(f"Using device: {device}, {num_workers} loading threads", flush=True)

    total_saved = 0
    failed_count = 0
    batch_idx = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]

        # Parallel image loading on CPU
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_image, batch_paths))

        # Filter out failed images and move to GPU
        batch_images = []
        batch_ids = []
        for img, img_id in results:
            if img is None:
                failed_count += 1
            else:
                batch_images.append(img.to(device))
                batch_ids.append(img_id)

        if not batch_images:
            continue

        # Stack and resize on GPU
        images = torch.stack(batch_images)
        images = K.geometry.transform.resize(
            images,
            (image_size, image_size),
            interpolation='bilinear',
            antialias=True
        )

        # Sanity check
        assert images.shape == (len(batch_images), 3, image_size, image_size), \
            f"Unexpected batch shape: {images.shape}"

        # Generate all rotations on GPU and save entire batch
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

            # Save entire batch as single file
            batch_data = {
                'images': rotated.cpu(),
                'ids': batch_ids
            }
            output_path = output_dir / f"batch_{batch_idx:04d}_rot{rotation_deg}.pt"
            torch.save(batch_data, output_path)
            total_saved += len(batch_ids)

        batch_idx += 1

    print(f"\nSaved {total_saved}/{total_expected} preprocessed tensors to {output_dir}", flush=True)
    if failed_count > 0:
        print(f"Failed to process {failed_count} images", flush=True)

    return total_saved


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated dataset preprocessing")
    parser.add_argument('--config', default='configs/h100.yaml', help='Config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    image_size = data_cfg.get('image_size', 256)

    preprocess_cfg = data_cfg.get('preprocessing', {})
    batch_size = preprocess_cfg.get('batch_size', 512)
    num_workers = preprocess_cfg.get('num_workers', 8)

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
        preprocess_split_gpu(
            split_path,
            output_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers
        )

    print("\nPreprocessing complete!", flush=True)


if __name__ == "__main__":
    main()
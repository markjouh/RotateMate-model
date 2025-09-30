#!/usr/bin/env python3
"""CPU-optimized dataset preprocessing with multiprocessing."""

import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .dataset import _gather_images


ROTATIONS = [0, 90, 180, 270]


def process_image(args):
    """Process a single image: load, resize, apply all rotations.

    Returns numpy arrays to avoid torch tensor pickling issues.
    """
    img_path, image_size = args

    try:
        # Load with OpenCV (SIMD-optimized)
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize with OpenCV (SIMD-optimized)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 in [0, 1] and CHW format
        img_np = img.astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))

        # Generate all rotations using numpy
        rotated_arrays = []
        for rotation_deg in ROTATIONS:
            if rotation_deg == 0:
                rotated = img_np
            elif rotation_deg == 90:
                rotated = np.rot90(img_np, k=1, axes=(1, 2))
            elif rotation_deg == 180:
                rotated = np.rot90(img_np, k=2, axes=(1, 2))
            elif rotation_deg == 270:
                rotated = np.rot90(img_np, k=3, axes=(1, 2))

            rotated_arrays.append(rotated.copy())

        return (img_path.stem, rotated_arrays)

    except Exception:
        return None


def preprocess_split_cpu(image_dir, output_dir, image_size=256, batch_size=4096, num_workers=16):
    """Preprocess images using CPU multiprocessing with OpenCV.

    Args:
        image_dir: Directory containing raw images
        output_dir: Directory to save preprocessed tensors
        image_size: Target image size (default 256)
        batch_size: Number of images per output file (default 4096)
        num_workers: Number of worker processes (default 16)

    Returns:
        Total number of preprocessed tensors saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _gather_images([image_dir])
    total_expected = len(image_paths) * len(ROTATIONS)

    print(f"Preprocessing {len(image_paths)} images with {len(ROTATIONS)} rotations", flush=True)
    print(f"Will save {total_expected} preprocessed tensors", flush=True)
    print(f"Using {num_workers} worker processes with OpenCV", flush=True)

    # Prepare arguments for parallel processing
    process_args = [(img_path, image_size) for img_path in image_paths]

    # Process images in parallel
    batch_idx = 0
    batch_data = {rot: {'images': [], 'ids': []} for rot in ROTATIONS}
    total_saved = 0
    failed_count = 0

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_image, process_args, chunksize=32),
                          total=len(image_paths), desc="Processing images"):
            if result is None:
                failed_count += 1
                continue

            img_id, rotated_arrays = result

            # Convert numpy arrays to torch tensors and add to batch
            for rot_idx, rotation_deg in enumerate(ROTATIONS):
                img_tensor = torch.from_numpy(rotated_arrays[rot_idx])
                batch_data[rotation_deg]['images'].append(img_tensor)
                batch_data[rotation_deg]['ids'].append(img_id)

            # Save batch when full
            if len(batch_data[ROTATIONS[0]]['images']) >= batch_size:
                for rotation_deg in ROTATIONS:
                    images_tensor = torch.stack(batch_data[rotation_deg]['images'])
                    output_path = output_dir / f"batch_{batch_idx:04d}_rot{rotation_deg}.pt"
                    torch.save({
                        'images': images_tensor,
                        'ids': batch_data[rotation_deg]['ids']
                    }, output_path)
                    total_saved += len(batch_data[rotation_deg]['ids'])

                batch_idx += 1
                batch_data = {rot: {'images': [], 'ids': []} for rot in ROTATIONS}

    # Save remaining images
    if batch_data[ROTATIONS[0]]['images']:
        for rotation_deg in ROTATIONS:
            images_tensor = torch.stack(batch_data[rotation_deg]['images'])
            output_path = output_dir / f"batch_{batch_idx:04d}_rot{rotation_deg}.pt"
            torch.save({
                'images': images_tensor,
                'ids': batch_data[rotation_deg]['ids']
            }, output_path)
            total_saved += len(batch_data[rotation_deg]['ids'])

    print(f"\nSaved {total_saved}/{total_expected} preprocessed tensors to {output_dir}", flush=True)
    if failed_count > 0:
        print(f"Failed to process {failed_count} images", flush=True)

    return total_saved


def main():
    parser = argparse.ArgumentParser(description="CPU-optimized dataset preprocessing with OpenCV")
    parser.add_argument('--config', default='configs/h100.yaml', help='Config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    image_size = data_cfg.get('image_size', 256)

    preprocess_cfg = data_cfg.get('preprocessing', {})
    batch_size = preprocess_cfg.get('batch_size', 4096)
    num_workers = preprocess_cfg.get('num_workers', 16)

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
        preprocess_split_cpu(
            split_path,
            output_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers
        )

    print("\nPreprocessing complete!", flush=True)


if __name__ == "__main__":
    main()
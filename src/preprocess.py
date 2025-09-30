#!/usr/bin/env python3
"""CPU-optimized dataset preprocessing with multiprocessing."""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image

from .dataset import _gather_images


ROTATIONS = [0, 90, 180, 270]


def process_image(args):
    """Process a single image: load, resize, apply all rotations."""
    img_path, image_size = args

    try:
        # Load and resize with PIL
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)

        # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
        img_tensor = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            .view(image_size, image_size, 3)
            .numpy()
        ).permute(2, 0, 1).float() / 255.0

        # Generate all rotations using torch.rot90
        rotated_tensors = []
        for rotation_deg in ROTATIONS:
            if rotation_deg == 0:
                rotated = img_tensor
            elif rotation_deg == 90:
                rotated = torch.rot90(img_tensor, k=1, dims=(1, 2))
            elif rotation_deg == 180:
                rotated = torch.rot90(img_tensor, k=2, dims=(1, 2))
            elif rotation_deg == 270:
                rotated = torch.rot90(img_tensor, k=3, dims=(1, 2))

            rotated_tensors.append(rotated)

        return (img_path.stem, rotated_tensors)

    except Exception:
        return None


def preprocess_split_cpu(image_dir, output_dir, image_size=256, batch_size=4096, num_workers=16):
    """Preprocess images using CPU multiprocessing.

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
    print(f"Using {num_workers} worker processes", flush=True)

    # Prepare arguments for parallel processing
    process_args = [(img_path, image_size) for img_path in image_paths]

    # Process images in parallel
    batch_idx = 0
    batch_data = {rot: {'images': [], 'ids': []} for rot in ROTATIONS}
    total_saved = 0
    failed_count = 0

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(process_image, process_args, chunksize=8),
                          total=len(image_paths), desc="Processing images"):
            if result is None:
                failed_count += 1
                continue

            img_id, rotated_tensors = result

            # Add to current batch for each rotation
            for rot_idx, rotation_deg in enumerate(ROTATIONS):
                batch_data[rotation_deg]['images'].append(rotated_tensors[rot_idx])
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
    parser = argparse.ArgumentParser(description="CPU-optimized dataset preprocessing")
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
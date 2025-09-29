"""
Processes images into rotated shards: applies 4 rotations (0°, 90°, 180°, 270°)
and saves as .pt files with letterboxing to 256x256.
"""

import logging
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def rotate_image(image, rotation_idx):
    if rotation_idx == 0:
        return image
    elif rotation_idx == 1:
        return image.transpose(Image.Transpose.ROTATE_90)
    elif rotation_idx == 2:
        return image.transpose(Image.Transpose.ROTATE_180)
    elif rotation_idx == 3:
        return image.transpose(Image.Transpose.ROTATE_270)
    else:
        raise ValueError(f"Invalid rotation index: {rotation_idx}")


def letterbox_resize(image, target_size=256, fill_color=(0, 0, 0)):
    w, h = image.size

    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), fill_color)

    if w >= h:
        new_w = target_size
        new_h = max(1, int(round(h * target_size / w)))
    else:
        new_h = target_size
        new_w = max(1, int(round(w * target_size / h)))

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))

    return canvas


def image_to_tensor(image):
    arr = np.asarray(image, dtype=np.uint8).copy()  # Make writable copy
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def process_single_image(image_path, rotations=[0, 90, 180, 270], target_size=256):
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            images = []
            labels = []

            for idx in range(4):
                if idx * 90 not in rotations:
                    continue

                rotated = rotate_image(img, idx)
                resized = letterbox_resize(rotated, target_size)
                tensor = image_to_tensor(resized)

                images.append(tensor)
                labels.append(idx)

            if images:
                return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)

    except Exception:
        pass

    return None


def process_image_batch(args):
    image_paths, rotations, target_size = args
    results = []

    for path in image_paths:
        result = process_single_image(path, rotations, target_size)
        if result is not None:
            results.append(result)

    return results


def create_shards(
    input_dirs,
    output_dir,
    samples_per_shard=5000,
    num_workers=8,
    rotations=[0, 90, 180, 270],
    target_size=256,
    batch_size=100
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning for images...")
    all_paths = []
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        for ext in IMAGE_EXTENSIONS:
            all_paths.extend(input_dir.glob(f"*{ext}"))
            all_paths.extend(input_dir.glob(f"*{ext.upper()}"))

    all_paths = sorted(set(all_paths))
    logger.info(f"Found {len(all_paths)} images")

    if not all_paths:
        raise ValueError("No images found")

    path_batches = []
    for i in range(0, len(all_paths), batch_size):
        batch = all_paths[i:i + batch_size]
        path_batches.append((batch, rotations, target_size))

    buffer_images = []
    buffer_labels = []
    shard_idx = 0
    total_samples = 0
    processed_images = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_image_batch, batch): batch
            for batch in path_batches
        }

        with tqdm(total=len(all_paths), desc="Processing images", unit="img") as pbar:
            for future in as_completed(future_to_batch):
                results = future.result()

                for img_tensors, labels in results:
                    buffer_images.append(img_tensors)
                    buffer_labels.append(labels)
                    processed_images += 1

                    current_samples = sum(t.size(0) for t in buffer_images)
                    if current_samples >= samples_per_shard:
                        all_images = torch.cat(buffer_images, 0)
                        all_labels = torch.cat(buffer_labels, 0)

                        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
                        torch.save({
                            "images_u8": all_images,
                            "labels": all_labels,
                            "rot_degs": torch.tensor(rotations)
                        }, shard_path)

                        logger.info(f"Wrote shard {shard_idx}: {all_images.shape[0]} samples")

                        total_samples += all_images.shape[0]
                        shard_idx += 1

                        buffer_images.clear()
                        buffer_labels.clear()

                batch_size_actual = len(future_to_batch[future][0])
                pbar.update(batch_size_actual)

    if buffer_images:
        all_images = torch.cat(buffer_images, 0)
        all_labels = torch.cat(buffer_labels, 0)

        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({
            "images_u8": all_images,
            "labels": all_labels,
            "rot_degs": torch.tensor(rotations)
        }, shard_path)

        logger.info(f"Wrote final shard {shard_idx}: {all_images.shape[0]} samples")
        total_samples += all_images.shape[0]
        shard_idx += 1

    stats = {
        "total_images": len(all_paths),
        "processed_images": processed_images,
        "failed_images": len(all_paths) - processed_images,
        "total_samples": total_samples,
        "shards_created": shard_idx,
        "output_dir": str(output_dir.resolve())
    }

    logger.info(f"Processing complete:")
    logger.info(f"  Processed: {stats['processed_images']}/{stats['total_images']} images")
    logger.info(f"  Created: {stats['shards_created']} shards")
    logger.info(f"  Total samples: {stats['total_samples']}")

    return stats


def verify_shards(shard_dir):
    shard_dir = Path(shard_dir)
    shard_files = sorted(shard_dir.glob("*.pt"))

    if not shard_files:
        raise ValueError(f"No shards found in {shard_dir}")

    total_samples = 0
    all_labels = []

    logger.info(f"Verifying {len(shard_files)} shards...")

    for shard_path in tqdm(shard_files, desc="Verifying"):
        data = torch.load(shard_path, map_location="cpu", weights_only=True)

        assert "images_u8" in data
        assert "labels" in data

        images = data["images_u8"]
        labels = data["labels"]
        assert images.shape[0] == labels.shape[0]
        assert images.dtype == torch.uint8
        assert labels.dtype == torch.int64

        total_samples += images.shape[0]
        all_labels.extend(labels.tolist())

    from collections import Counter
    label_counts = Counter(all_labels)

    return {
        "num_shards": len(shard_files),
        "total_samples": total_samples,
        "label_distribution": dict(label_counts)
    }
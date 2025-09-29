"""Rotation classification dataset with on-the-fly preprocessing."""

import logging
from contextlib import suppress
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _gather_images(paths):
    """Collect all image files from given paths (files or directories)."""
    image_paths = []
    for entry in paths:
        path = Path(entry)
        if not path.exists():
            logger.warning("Path does not exist: %s", path)
            continue

        if path.is_file():
            image_paths.append(path)
        else:
            # Gather all images with supported extensions (case-insensitive)
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(path.glob(f"*{ext}"))
                image_paths.extend(path.glob(f"*{ext.upper()}"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {paths}")

    return sorted(set(p.resolve() for p in image_paths))


def letterbox_resize(image, target_size, fill_color=(0, 0, 0)):
    """Resize image preserving aspect ratio, padding to square."""
    width, height = image.size
    if width == 0 or height == 0:
        return Image.new("RGB", (target_size, target_size), fill_color)

    # Calculate new dimensions (longest side = target_size)
    scale = target_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # Resize and center on canvas
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    offset = ((target_size - new_width) // 2, (target_size - new_height) // 2)
    canvas.paste(resized, offset)
    return canvas


class RotationDataset(Dataset):
    """Dataset that generates rotation classification samples on-the-fly."""

    def __init__(self, image_dirs, rotations, image_size, transform=None, max_images=None, fill_color=(0, 0, 0)):
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]

        self.image_paths = _gather_images(image_dirs)
        if max_images:
            self.image_paths = self.image_paths[:max_images]

        self.rotations = list(rotations)
        if not self.rotations:
            raise ValueError("Must provide at least one rotation angle")

        self.image_size = image_size
        self.fill_color = fill_color
        self.transform = transform or T.ToTensor()

        logger.info(
            "Loaded %d images x %d rotations = %d samples",
            len(self.image_paths), len(self.rotations), len(self)
        )

    def __len__(self):
        return len(self.image_paths) * len(self.rotations)

    def __getitem__(self, index):
        image_idx, rotation_idx = divmod(index, len(self.rotations))
        rotation_deg = self.rotations[rotation_idx]

        # Load and preprocess image
        img = Image.open(self.image_paths[image_idx])
        with suppress(Exception):
            img = ImageOps.exif_transpose(img)  # Fix EXIF orientation
        img = img.convert("RGB")
        img = F.rotate(img, rotation_deg, fill=self.fill_color)
        img = letterbox_resize(img, self.image_size, self.fill_color)
        return self.transform(img), rotation_idx


def create_dataloaders(
    train_dir,
    val_dir,
    test_dir=None,
    rotations=(0, 90, 180, 270),
    image_size=256,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    max_train_images=None,
    max_val_images=None,
    max_test_images=None,
):
    """Create train/val/test dataloaders for rotation classification."""

    # Create datasets
    train_dataset = RotationDataset(train_dir, rotations, image_size, max_images=max_train_images)
    val_dataset = RotationDataset(val_dir, rotations, image_size, max_images=max_val_images)

    # DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloaders = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    }

    # Optional test set
    if test_dir and Path(test_dir).exists():
        test_dataset = RotationDataset(test_dir, rotations, image_size, max_images=max_test_images)
        dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    logger.info("Created dataloaders: %s", list(dataloaders.keys()))
    for name, loader in dataloaders.items():
        logger.info("  %s: %d samples, %d batches", name, len(loader.dataset), len(loader))

    return dataloaders

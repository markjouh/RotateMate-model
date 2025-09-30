"""Rotation classification dataset."""

import logging
from pathlib import Path
import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _gather_images(paths):
    """Collect all image files from given paths."""
    image_paths = []
    for entry in paths:
        path = Path(entry)
        if not path.exists():
            logger.warning("Path does not exist: %s", path)
            continue

        if path.is_file():
            image_paths.append(path)
        else:
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(path.glob(f"*{ext}"))
                image_paths.extend(path.glob(f"*{ext.upper()}"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {paths}")

    return sorted(set(p.resolve() for p in image_paths))


class RotationDataset(Dataset):
    """Dataset that generates rotation classification samples on-the-fly."""

    def __init__(self, image_dir, image_size=256, max_images=None):
        self.image_paths = _gather_images([image_dir])
        if max_images:
            self.image_paths = self.image_paths[:max_images]

        self.image_size = image_size
        self.rotations = [0, 90, 180, 270]

        logger.info("Created dataset with %d images", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths) * len(self.rotations)

    def __getitem__(self, index):
        # Map index to (image_index, rotation_index)
        img_idx = index // len(self.rotations)
        rot_idx = index % len(self.rotations)

        # Load image
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert('RGB')

        # Resize
        img = TF.resize(img, (self.image_size, self.image_size))

        # Apply rotation
        rotation_deg = self.rotations[rot_idx]
        if rotation_deg != 0:
            img = TF.rotate(img, rotation_deg)

        # To tensor and normalize
        img = TF.to_tensor(img)

        return img, rot_idx


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
    """Create train/val/test dataloaders."""

    train_dataset = RotationDataset(train_dir, image_size, max_train_images)
    val_dataset = RotationDataset(val_dir, image_size, max_val_images)

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

    if test_dir:
        test_dataset = RotationDataset(test_dir, image_size, max_test_images)
        dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    logger.info("Created dataloaders: %s", list(dataloaders.keys()))
    for name, loader in dataloaders.items():
        logger.info("  %s: %d samples, %d batches", name, len(loader.dataset), len(loader))

    return dataloaders
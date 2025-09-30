"""Rotation classification dataset using preprocessed tensors."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

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


class PreprocessedRotationDataset(Dataset):
    """Fast dataset that loads preprocessed batches."""

    def __init__(self, preprocessed_dir, rotations, max_images=None):
        self.preprocessed_dir = Path(preprocessed_dir)
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

        self.rotations = list(rotations)

        # Load all batch files and build index
        batch_files = sorted(self.preprocessed_dir.glob("batch_*.pt"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {preprocessed_dir}")

        self.samples = []
        for batch_file in batch_files:
            batch_data = torch.load(batch_file, weights_only=True)
            images = batch_data['images']

            # Extract rotation from filename
            rotation_str = batch_file.stem.split("_rot")[1]
            rotation_deg = int(rotation_str)
            rotation_idx = self.rotations.index(rotation_deg)

            # Add each image in batch to samples list
            for i in range(len(images)):
                self.samples.append((batch_file, i, rotation_idx, images[i]))

        if max_images and len(self.samples) > max_images:
            self.samples = self.samples[:max_images]

        logger.info("Loaded %d preprocessed samples from %d batch files",
                   len(self.samples), len(batch_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, _, rotation_idx, image = self.samples[index]
        return image, rotation_idx


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
    """Create train/val/test dataloaders for rotation classification.

    Note: rotations parameter is kept for compatibility but always uses [0, 90, 180, 270].
    """
    rotations = [0, 90, 180, 270]

    # Expect preprocessed data
    train_preprocessed = Path(str(train_dir) + "_preprocessed")
    val_preprocessed = Path(str(val_dir) + "_preprocessed")

    train_dataset = PreprocessedRotationDataset(train_preprocessed, rotations, max_train_images)
    val_dataset = PreprocessedRotationDataset(val_preprocessed, rotations, max_val_images)

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
    if test_dir:
        test_preprocessed = Path(str(test_dir) + "_preprocessed")
        if test_preprocessed.exists():
            test_dataset = PreprocessedRotationDataset(test_preprocessed, rotations, max_test_images)
            dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    logger.info("Created dataloaders: %s", list(dataloaders.keys()))
    for name, loader in dataloaders.items():
        logger.info("  %s: %d samples, %d batches", name, len(loader.dataset), len(loader))

    return dataloaders
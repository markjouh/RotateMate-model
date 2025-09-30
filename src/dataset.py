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
    """Fast dataset that loads preprocessed tensor files."""

    def __init__(self, preprocessed_dir, rotations, max_images=None):
        self.preprocessed_dir = Path(preprocessed_dir)
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

        self.rotations = list(rotations)
        self.tensor_paths = sorted(self.preprocessed_dir.glob("*.pt"))

        if max_images:
            self.tensor_paths = self.tensor_paths[:max_images * len(rotations)]

        if not self.tensor_paths:
            raise FileNotFoundError(f"No .pt files found in {preprocessed_dir}")

        logger.info("Loaded %d preprocessed samples", len(self.tensor_paths))

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, index):
        tensor = torch.load(self.tensor_paths[index], weights_only=True)
        rotation_str = self.tensor_paths[index].stem.split("_rot")[1]
        rotation_idx = self.rotations.index(int(rotation_str))
        return tensor, rotation_idx


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

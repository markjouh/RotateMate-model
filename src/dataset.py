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
    """Fast dataset that loads preprocessed batches lazily."""

    def __init__(self, preprocessed_dir, rotations, max_images=None):
        self.preprocessed_dir = Path(preprocessed_dir)
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

        self.rotations = list(rotations)
        self.batch_cache = {}

        # Build lightweight index: just store batch file paths grouped by rotation
        batch_files = sorted(self.preprocessed_dir.glob("batch_*.pt"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {preprocessed_dir}")

        # Group batch files by rotation
        self.rotation_batches = {rot: [] for rot in rotations}
        for batch_file in batch_files:
            rotation_str = batch_file.stem.split("_rot")[1]
            rotation_deg = int(rotation_str)
            self.rotation_batches[rotation_deg].append(batch_file)

        # Lazy load: build samples list on first access
        self._samples = None
        self._indexed = False

    def _build_index(self):
        """Build the samples index lazily on first access."""
        if self._indexed:
            return

        self._samples = []
        for rotation_deg, batch_files in self.rotation_batches.items():
            rotation_idx = self.rotations.index(rotation_deg)
            for batch_file in batch_files:
                # Use mmap to get size without loading
                data = torch.load(batch_file, weights_only=True, mmap=True)
                num_images = len(data['images'])
                del data

                for i in range(num_images):
                    self._samples.append((batch_file, i, rotation_idx))

        self._indexed = True
        logger.info("Indexed %d samples", len(self._samples))

    def __len__(self):
        if not self._indexed:
            self._build_index()
        return len(self._samples)

    def __getitem__(self, index):
        if not self._indexed:
            self._build_index()

        batch_file, batch_idx, rotation_idx = self._samples[index]

        # Load batch if not cached
        if batch_file not in self.batch_cache:
            batch_data = torch.load(batch_file, weights_only=True)
            self.batch_cache[batch_file] = batch_data['images']

            # LRU: keep cache size reasonable
            if len(self.batch_cache) > 8:
                oldest = next(iter(self.batch_cache))
                del self.batch_cache[oldest]

        image = self.batch_cache[batch_file][batch_idx]
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
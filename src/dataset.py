"""Rotation classification dataset with on-the-fly preprocessing."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic")


def _gather_images(paths: Sequence[Path | str]) -> List[Path]:
    image_paths: List[Path] = []
    for entry in paths:
        directory = Path(entry)
        if not directory.exists():
            logger.warning("Image directory missing: %s", directory)
            continue

        if directory.is_file():
            image_paths.append(directory)
            continue

        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {paths}")

    return sorted({path.resolve() for path in image_paths})


class RotationDataset(Dataset):
    def __init__(
        self,
        image_dirs: Sequence[Path | str],
        rotations: Sequence[int],
        transform: Optional[T.Compose] = None,
        max_images: Optional[int] = None,
    ) -> None:
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]

        self.image_paths = _gather_images(image_dirs)
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        self.rotations = list(rotations)
        if not self.rotations:
            raise ValueError("At least one rotation must be provided")

        self.samples_per_image = len(self.rotations)
        self.transform = transform or T.Compose([
            T.Resize(256, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
        ])

        logger.info(
            "Loaded %d base images producing %d samples per epoch",
            len(self.image_paths),
            len(self.image_paths) * self.samples_per_image,
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, index: int):
        image_idx, rotation_idx = divmod(index, self.samples_per_image)
        path = self.image_paths[image_idx]
        rotation_deg = self.rotations[rotation_idx]

        with Image.open(path) as img:
            with suppress(Exception):
                img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = F.rotate(img, rotation_deg, interpolation=InterpolationMode.BILINEAR)

        tensor = self.transform(img)
        return tensor, rotation_idx


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])


def create_dataloaders(
    train_dir: Path | str,
    val_dir: Path | str,
    *,
    test_dir: Optional[Path | str] = None,
    rotations: Sequence[int] = (0, 90, 180, 270),
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    max_test_images: Optional[int] = None,
) -> dict[str, DataLoader]:
    transform = _build_transform(image_size)

    train_dataset = RotationDataset([train_dir], rotations, transform, max_train_images)
    val_dataset = RotationDataset([val_dir], rotations, transform, max_val_images)

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloaders: dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    }

    if test_dir is not None and Path(test_dir).exists():
        test_dataset = RotationDataset([test_dir], rotations, transform, max_test_images)
        dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    for name, loader in dataloaders.items():
        logger.info("%s loader: %d samples, %d batches", name, len(loader.dataset), len(loader))

    return dataloaders

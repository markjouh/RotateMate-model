"""Rotation classification dataset with on-the-fly preprocessing."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageOps
import torch
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


def letterbox_resize(image: Image.Image, target_size: int, fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    width, height = image.size
    if width == 0 or height == 0:
        return Image.new("RGB", (target_size, target_size), fill_color)

    if width >= height:
        new_width = target_size
        new_height = max(1, int(round(height * target_size / width)))
    else:
        new_height = target_size
        new_width = max(1, int(round(width * target_size / height)))

    resized = image.resize((new_width, new_height), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    canvas.paste(resized, ((target_size - new_width) // 2, (target_size - new_height) // 2))
    return canvas


class RotationDataset(Dataset):
    def __init__(
        self,
        image_dirs: Sequence[Path | str],
        rotations: Sequence[int],
        *,
        image_size: int,
        transform: Optional[T.Compose] = None,
        max_images: Optional[int] = None,
        fill_color: tuple[int, int, int] = (0, 0, 0),
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
        self.image_size = image_size
        self.fill_color = fill_color
        self.transform = transform or T.ToTensor()

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
            rotated = F.rotate(
                img,
                rotation_deg,
                interpolation=InterpolationMode.BILINEAR,
                fill=self.fill_color,
            )

        letterboxed = letterbox_resize(rotated, self.image_size, self.fill_color)
        tensor = self.transform(letterboxed)
        return tensor, rotation_idx


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
    transform = T.ToTensor()

    train_dataset = RotationDataset(
        [train_dir],
        rotations,
        image_size=image_size,
        transform=transform,
        max_images=max_train_images,
    )
    val_dataset = RotationDataset(
        [val_dir],
        rotations,
        image_size=image_size,
        transform=transform,
        max_images=max_val_images,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if pin_memory and torch.cuda.is_available():
        loader_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloaders: dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    }

    if test_dir is not None and Path(test_dir).exists():
        test_dataset = RotationDataset(
            [test_dir],
            rotations,
            image_size=image_size,
            transform=transform,
            max_images=max_test_images,
        )
        dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    for name, loader in dataloaders.items():
        logger.info("%s loader: %d samples, %d batches", name, len(loader.dataset), len(loader))

    return dataloaders

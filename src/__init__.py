"""RotateMate model training package."""

__version__ = "0.1.0"

from .dataset import RotationDataset, create_dataloaders
from .trainer import Trainer
from .exporter import export_model
from .downloader import download_and_extract, verify_dataset

__all__ = [
    "RotationDataset",
    "create_dataloaders",
    "Trainer",
    "export_model",
    "download_and_extract",
    "verify_dataset",
]
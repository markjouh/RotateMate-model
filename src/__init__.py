"""RotateMate model training package."""

__version__ = "0.1.0"

from .dataset import PreprocessedRotationDataset, create_dataloaders
from .trainer import Trainer, ModelWrapper
from .exporter import export_model
from .downloader import download_and_extract, verify_dataset

__all__ = [
    "PreprocessedRotationDataset",
    "create_dataloaders",
    "Trainer",
    "ModelWrapper",
    "export_model",
    "download_and_extract",
    "verify_dataset",
]
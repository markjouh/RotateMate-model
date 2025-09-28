import bisect
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ShardedDataset(Dataset):
    def __init__(self, shard_dirs, cache_size=3, transforms=None):
        if isinstance(shard_dirs, (str, Path)):
            shard_dirs = [shard_dirs]

        self.files = []
        for shard_dir in shard_dirs:
            shard_dir = Path(shard_dir)
            if shard_dir.is_dir():
                self.files.extend(sorted(shard_dir.glob("*.pt")))
            elif shard_dir.exists():
                self.files.append(shard_dir)

        if not self.files:
            raise FileNotFoundError(f"No .pt shards found in {shard_dirs}")

        logger.info(f"Found {len(self.files)} shard files")

        self.cumulative_sizes = []
        total_samples = 0

        for filepath in self.files:
            data = torch.load(filepath, map_location="cpu", weights_only=True)
            num_samples = data["labels"].numel()
            total_samples += num_samples
            self.cumulative_sizes.append(total_samples)

        self.total_samples = total_samples
        self.cache_size = cache_size
        self.cache = {}
        self.transforms = transforms

    def __len__(self):
        return self.total_samples

    def _get_shard_and_index(self, global_idx):
        shard_idx = bisect.bisect_left(self.cumulative_sizes, global_idx + 1)

        if shard_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[shard_idx - 1]

        return shard_idx, local_idx

    def _load_shard(self, shard_idx):
        if shard_idx in self.cache:
            return self.cache[shard_idx]

        shard_data = torch.load(
            self.files[shard_idx],
            map_location="cpu",
            weights_only=True
        )

        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.cache.keys())
            del self.cache[oldest_key]

        self.cache[shard_idx] = shard_data
        return shard_data

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        shard_idx, local_idx = self._get_shard_and_index(idx)
        shard_data = self._load_shard(shard_idx)

        image = shard_data["images_u8"][local_idx].float().div_(255.0)
        label = shard_data["labels"][local_idx].long()

        if self.transforms:
            image = self.transforms(image)

        return image, label


def create_dataloaders(
    train_dir,
    val_dir,
    test_dir=None,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    transforms=None
):
    dataloaders = {}

    train_dataset = ShardedDataset(train_dir, transforms=transforms)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    val_dataset = ShardedDataset(val_dir, transforms=transforms)
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    if test_dir and Path(test_dir).exists():
        test_dataset = ShardedDataset(test_dir, transforms=transforms)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    logger.info(f"Created dataloaders: {list(dataloaders.keys())}")
    for name, loader in dataloaders.items():
        logger.info(f"  {name}: {len(loader.dataset)} samples, {len(loader)} batches")

    return dataloaders
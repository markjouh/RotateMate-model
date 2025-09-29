"""
Downloads and extracts COCO dataset with progress bars and resume support.
"""

import os
import zipfile
import logging
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    filepath = dest_path / filename

    if filepath.exists():
        logger.info(f"Already exists: {filepath}")
        return filepath

    logger.info(f"Downloading {url}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urlretrieve(url, filepath, reporthook=t.update_to)

    return filepath


def extract_zip(zip_path, extract_to):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    # Check if extraction folder already has images
    split_name = zip_path.stem  # e.g., "train2017", "val2017", "test2017"
    target_folder = extract_to / split_name

    if target_folder.exists():
        existing_images = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.png"))
        if existing_images:
            logger.info(f"Already extracted: {split_name} ({len(existing_images)} images found)")
            return extract_to

    logger.info(f"Extracting {zip_path.name}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_size = sum(f.file_size for f in zip_ref.filelist)

        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
            for file in zip_ref.filelist:
                zip_ref.extract(file, extract_to)
                pbar.update(file.file_size)

    return extract_to


def download_and_extract(urls, raw_dir, extract_dir):
    raw_dir = Path(raw_dir)
    extract_dir = Path(extract_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        zip_path = download_file(url, raw_dir)
        # Extract directly to extract_dir, COCO zips already contain split folders
        extract_zip(zip_path, extract_dir)


def verify_dataset(extract_dir):
    extract_dir = Path(extract_dir)
    splits = {}

    for split in ['train2017', 'val2017', 'test2017']:
        split_path = extract_dir / split
        if split_path.exists():
            image_count = len(list(split_path.glob("*.jpg")))
            logger.info(f"Found {split}: {image_count} images")
            splits[split] = split_path

    if not splits:
        raise ValueError(f"No splits found in {extract_dir}")

    return splits

"""Downloads and extracts COCO dataset with progress bars and resume support."""

import logging
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url, dest_path):
    """Download file with progress bar, skip if already exists."""
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    filename = Path(urlparse(url).path).name
    filepath = dest_path / filename

    if filepath.exists():
        logger.info("Already downloaded: %s", filename)
        return filepath

    logger.info("Downloading: %s", url)

    # Progress bar callback
    pbar = tqdm(unit='B', unit_scale=True, desc=filename)

    def update_progress(blocks, block_size, total_size):
        if pbar.total is None and total_size > 0:
            pbar.total = total_size
        pbar.update(blocks * block_size - pbar.n)

    temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
    try:
        urlretrieve(url, temp_path, reporthook=update_progress)
        temp_path.rename(filepath)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    finally:
        pbar.close()

    return filepath


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar, skip if already extracted."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    split_name = zip_path.stem  # e.g., "train2017"
    target_folder = extract_to / split_name

    if target_folder.exists():
        image_count = len(list(target_folder.glob("*.jpg")))
        if image_count > 0:
            logger.info("Already extracted: %s (%d images)", split_name, image_count)
            return extract_to

    logger.info("Extracting: %s", zip_path.name)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        total_size = sum(f.file_size for f in zf.filelist)

        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
            for member in zf.filelist:
                zf.extract(member, extract_to)
                pbar.update(member.file_size)

    return extract_to


def download_and_extract(urls, raw_dir, extract_dir):
    """Download and extract all dataset URLs."""
    raw_dir = Path(raw_dir)
    extract_dir = Path(extract_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        zip_path = download_file(url, raw_dir)
        extract_zip(zip_path, extract_dir)


def verify_dataset(extract_dir):
    """Verify dataset directory and return available splits."""
    extract_dir = Path(extract_dir)
    splits = {}

    for split_name in ['train2017', 'val2017', 'test2017']:
        split_path = extract_dir / split_name
        if split_path.exists():
            image_count = len(list(split_path.glob("*.jpg")))
            if image_count > 0:
                logger.info("Found %s: %d images", split_name, image_count)
                splits[split_name] = split_path

    if not splits:
        raise ValueError(f"No dataset splits found in {extract_dir}")

    return splits

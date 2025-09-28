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


def download_file(url, dest_path, force=False):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    filepath = dest_path.parent / filename

    if filepath.exists() and not force:
        logger.info(f"File already exists: {filepath}")
        return filepath

    logger.info(f"Downloading {url}")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urlretrieve(url, filepath, reporthook=t.update_to)
        return filepath
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise


def extract_zip(zip_path, extract_to, remove_zip=False):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    marker_file = extract_to / f".extracted_{zip_path.stem}"
    if marker_file.exists():
        logger.info(f"Already extracted: {zip_path.name}")
        return extract_to

    logger.info(f"Extracting {zip_path.name}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_size = sum(f.file_size for f in zip_ref.filelist)

        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Extracting") as pbar:
            for file in zip_ref.filelist:
                zip_ref.extract(file, extract_to)
                pbar.update(file.file_size)

    marker_file.touch()

    if remove_zip:
        zip_path.unlink()

    return extract_to


def download_and_extract(urls, raw_dir, extract_dir, remove_zips=False):
    raw_dir = Path(raw_dir)
    extract_dir = Path(extract_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_dirs = []

    for url in urls:
        zip_path = download_file(url, raw_dir)
        dataset_name = Path(urlparse(url).path).stem
        extract_to = extract_dir / dataset_name
        extract_path = extract_zip(zip_path, extract_to, remove_zips)
        extracted_dirs.append(extract_path)

    return extracted_dirs


def verify_dataset(extract_dir, expected_splits=['train2017', 'val2017', 'test2017']):
    extract_dir = Path(extract_dir)
    splits = {}

    for split in expected_splits:
        split_path = extract_dir / split
        if split_path.exists():
            image_count = len(list(split_path.glob("*.jpg"))) + len(list(split_path.glob("*.png")))
            logger.info(f"Found {split}: {image_count} images")
            splits[split] = split_path

    if not splits:
        raise ValueError(f"No valid splits found in {extract_dir}")

    return splits
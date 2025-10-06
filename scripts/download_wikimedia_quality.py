#!/usr/bin/env python3
"""
Download Wikipedia Commons Quality Images and preprocess them.
- Downloads images from Category:Quality_images (~410k images)
- Scales to 224px on long side
- Letterboxes to 224x224 with black bars
- Saves as JPG
"""

import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "images"
TARGET_SIZE = 224
MAX_WORKERS = 64
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    'User-Agent': 'RotateMate-Dataset-Downloader/1.0 (Educational ML project; contact: user@example.com)'
}

def get_image_batch_with_urls(continue_token=None):
    """Get a batch of Quality Images with their URLs in one API call."""
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'categorymembers',
        'gcmtitle': 'Category:Quality_images',
        'gcmlimit': 50,
        'gcmtype': 'file',
        'prop': 'imageinfo',
        'iiprop': 'url',
    }

    if continue_token:
        params['gcmcontinue'] = continue_token

    response = requests.get(API_URL, params=params, headers=HEADERS)

    if response.status_code != 200:
        return [], None

    try:
        data = response.json()
    except:
        return [], None

    urls = []
    if 'query' in data and 'pages' in data['query']:
        for page in data['query']['pages'].values():
            if 'imageinfo' in page and page['imageinfo']:
                url = page['imageinfo'][0].get('url')
                if url:
                    urls.append(url)

    next_token = data.get('continue', {}).get('gcmcontinue')
    return urls, next_token

def preprocess_image(img):
    """
    Resize image so long side is 224px, then letterbox to 224x224 with black bars.
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Calculate scaling to make long side 224px
    width, height = img.size
    if width > height:
        new_width = TARGET_SIZE
        new_height = int(height * (TARGET_SIZE / width))
    else:
        new_height = TARGET_SIZE
        new_width = int(width * (TARGET_SIZE / height))

    # Resize
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black 224x224 canvas and paste resized image centered
    canvas = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    x_offset = (TARGET_SIZE - new_width) // 2
    y_offset = (TARGET_SIZE - new_height) // 2
    canvas.paste(img, (x_offset, y_offset))

    return canvas

def download_and_process_image(url, index):
    """Download a single image, preprocess it, and save as JPG."""
    try:
        response = requests.get(url, timeout=30, headers=HEADERS)
        if response.status_code != 200:
            return False

        img = Image.open(BytesIO(response.content))
        processed = preprocess_image(img)

        output_path = OUTPUT_DIR / f"{index}.jpg"
        processed.save(output_path, 'JPEG', quality=85)

        return True

    except:
        return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {OUTPUT_DIR}")

    success = 0
    errors = 0
    total = 0
    continue_token = None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        pbar = tqdm(unit="img")

        while True:
            urls, continue_token = get_image_batch_with_urls(continue_token)

            for url in urls:
                future = executor.submit(download_and_process_image, url, total)
                futures[future] = total
                total += 1

            done = [f for f in futures if f.done()]
            for f in done:
                if f.result():
                    success += 1
                else:
                    errors += 1
                pbar.update(1)
                pbar.set_postfix({'ok': success, 'err': errors})
                del futures[f]

            if not continue_token:
                break

        for f in as_completed(futures):
            if f.result():
                success += 1
            else:
                errors += 1
            pbar.update(1)
            pbar.set_postfix({'ok': success, 'err': errors})

        pbar.close()

    print(f"\nDone! {success} ok, {errors} errors, {total} total")

if __name__ == "__main__":
    main()

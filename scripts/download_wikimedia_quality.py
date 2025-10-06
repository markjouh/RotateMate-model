#!/Users/mark/RotateMate-model/venv/bin/python
"""
Download Wikipedia Commons Quality Images and preprocess them.
- Downloads images from Category:Quality_images (~410k images)
- Scales to 224px on long side
- Letterboxes to 224x224 with black bars
- Saves as JPG
"""

import os
import sys
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Configuration
OUTPUT_DIR = Path("/Users/mark/RotateMate-model/images")
TARGET_SIZE = 224
MAX_WORKERS = 12
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    'User-Agent': 'RotateMate-Dataset-Downloader/1.0 (Educational ML project; contact: user@example.com)'
}

def get_quality_images_list():
    """Get list of all Quality Images from Wikimedia Commons using the API."""
    images = []
    continue_token = None

    print("Fetching list of Quality Images from Wikimedia Commons...")

    while True:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': 'Category:Quality_images',
            'cmlimit': 500,  # Max allowed
            'cmtype': 'file',
            'cmprop': 'title'
        }

        if continue_token:
            params['cmcontinue'] = continue_token

        response = requests.get(API_URL, params=params, headers=HEADERS)

        # Debug the response
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response text: {response.text[:500]}")
            sys.exit(1)

        try:
            data = response.json()
        except Exception as e:
            print(f"Failed to parse JSON. Response text: {response.text[:500]}")
            raise

        if 'query' in data and 'categorymembers' in data['query']:
            batch = data['query']['categorymembers']
            images.extend([item['title'] for item in batch])
            print(f"Fetched {len(images)} images so far...")

        if 'continue' in data and 'cmcontinue' in data['continue']:
            continue_token = data['continue']['cmcontinue']
            time.sleep(0.01)  # Be nice to the API
        else:
            break

    print(f"Total Quality Images found: {len(images)}")
    return images

def get_image_url(filename):
    """Get the direct URL for an image file."""
    params = {
        'action': 'query',
        'format': 'json',
        'titles': filename,
        'prop': 'imageinfo',
        'iiprop': 'url'
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    pages = data.get('query', {}).get('pages', {})
    for page_id, page_data in pages.items():
        if 'imageinfo' in page_data and len(page_data['imageinfo']) > 0:
            return page_data['imageinfo'][0].get('url')

    return None

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

def download_and_process_image(filename, index):
    """Download a single image, preprocess it, and save as JPG."""
    try:
        # Get image URL
        url = get_image_url(filename)
        if not url:
            return False, f"No URL found for {filename}"

        # Download image
        response = requests.get(url, timeout=30, headers=HEADERS)
        if response.status_code != 200:
            return False, f"Failed to download {filename}: HTTP {response.status_code}"

        # Open and preprocess
        img = Image.open(BytesIO(response.content))
        processed = preprocess_image(img)

        # Save as JPG
        output_path = OUTPUT_DIR / f"{index}.jpg"
        processed.save(output_path, 'JPEG', quality=95)

        return True, filename

    except Exception as e:
        return False, f"Error processing {filename}: {str(e)}"

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of all Quality Images
    image_list = get_quality_images_list()

    if not image_list:
        print("No images found!")
        return

    print(f"\nDownloading and processing {len(image_list)} images...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Download and process images in parallel
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_and_process_image, filename, idx): (filename, idx)
            for idx, filename in enumerate(image_list)
        }

        with tqdm(total=len(image_list), desc="Processing") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # Only print first 10 errors
                        print(f"\n{message}")

                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'errors': error_count})

    print(f"\n✓ Download complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    main()

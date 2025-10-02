import argparse
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from train import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Find model failures')
    parser.add_argument('--checkpoint', type=str, default='rotation_model.pth', help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--workers', type=int, default=16, help='Number of data loader workers')
    parser.add_argument('--output-dir', type=str, default='failures', help='Output directory for failed images')
    return parser.parse_args()


def find_failures(model, loader, device, dataset_name, output_dir, batch_size):
    """Find and save images where the model prediction is incorrect."""
    model.eval()
    failures = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=f"Testing {dataset_name}")):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)

            # Find incorrect predictions
            incorrect = predicted != labels
            incorrect_indices = incorrect.nonzero(as_tuple=True)[0]

            # Store failure information
            for idx in incorrect_indices:
                global_idx = batch_idx * batch_size + idx.item()
                failures.append({
                    'idx': global_idx,
                    'true_label': labels[idx].item(),
                    'pred_label': predicted[idx].item()
                })

    # Save failed images to output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    for failure in failures:
        img_path = loader.dataset.img_paths[failure['idx']]
        true_rot = failure['true_label']
        pred_rot = failure['pred_label']

        # Copy image with descriptive name
        img_name = Path(img_path).stem
        output_name = f"{img_name}_true{true_rot}_pred{pred_rot}.jpg"
        shutil.copy(img_path, dataset_output_dir / output_name)

    return failures


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 224

    # Load model
    model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    print(f"Loaded model from {args.checkpoint}")

    # Datasets
    train_dataset = Dataset("data/train2017", img_size=img_size)
    val_dataset = Dataset("data/val2017", img_size=img_size)

    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.workers, pin_memory=True)

    # Find failures
    print(f"\nFinding failures on train2017...")
    train_failures = find_failures(model, train_loader, device, "train2017", args.output_dir, args.batch_size)
    print(f"Found {len(train_failures)} failures on train2017 ({100.0 * len(train_failures) / len(train_dataset):.2f}%)")

    print(f"\nFinding failures on val2017...")
    val_failures = find_failures(model, val_loader, device, "val2017", args.output_dir, args.batch_size)
    print(f"Found {len(val_failures)} failures on val2017 ({100.0 * len(val_failures) / len(val_dataset):.2f}%)")

    print(f"\nFailed images saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

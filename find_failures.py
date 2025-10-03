import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import timm
from tqdm import tqdm

from train import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Find model failures')
    parser.add_argument('--checkpoint', type=str, default='rotation_model.pth', help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--workers', type=int, default=16, help='Number of data loader workers')
    parser.add_argument('--output-dir', type=str, default='failures', help='Output directory for failed images')
    parser.add_argument('--max-failures', type=int, default=500, help='Maximum number of failures to save per dataset')
    return parser.parse_args()


def find_failures(model, loader, device, dataset_name, output_dir, max_failures=None):
    """Find and save images where the model prediction is incorrect."""
    model.eval()
    num_failures = 0

    # ImageNet normalization params for denormalization
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Save failed images to output directory
    dataset_output_dir = Path(output_dir) / dataset_name

    # Clear existing failures
    if dataset_output_dir.exists():
        for f in dataset_output_dir.glob("*.png"):
            f.unlink()

    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Testing {dataset_name}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)

            # Find incorrect predictions
            incorrect = predicted != labels
            incorrect_indices = incorrect.nonzero(as_tuple=True)[0]

            # Save failure images
            for idx in incorrect_indices:
                num_failures += 1
                true_rot = labels[idx].item()
                pred_rot = predicted[idx].item()

                # Get confidence scores
                probs = torch.softmax(outputs[idx], dim=0)
                pred_conf = probs[pred_rot].item()
                true_conf = probs[true_rot].item()

                # Denormalize the processed image
                img_tensor = imgs[idx].cpu()
                img_denorm = img_tensor * IMAGENET_STD + IMAGENET_MEAN
                img_denorm = img_denorm.clamp(0, 1)

                # Save processed image with descriptive name
                global_idx = sample_idx + idx.item()
                img_path = loader.dataset.img_paths[global_idx]
                img_name = Path(img_path).stem
                output_name = f"{img_name}_true{true_rot}_{true_conf:.3f}_pred{pred_rot}_{pred_conf:.3f}.png"
                save_image(img_denorm, dataset_output_dir / output_name)

                # Stop if we've saved max_failures
                if max_failures is not None and num_failures >= max_failures:
                    return num_failures

            sample_idx += len(imgs)

    return num_failures


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
    train_dataset = Dataset("data/train2017", img_size=img_size, augment=True, fixed_rotation=False)
    val_dataset = Dataset("data/val2017", img_size=img_size, augment=False, fixed_rotation=True)

    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.workers, pin_memory=True)

    # Find failures
    print(f"\nFinding failures on train2017 (with augmentation, max {args.max_failures} failures)...")
    train_failures = find_failures(model, train_loader, device, "train2017", args.output_dir, max_failures=args.max_failures)
    print(f"Found {train_failures} failures on train2017")

    print(f"\nFinding failures on val2017 (max {args.max_failures} failures)...")
    val_failures = find_failures(model, val_loader, device, "val2017", args.output_dir, max_failures=args.max_failures)
    print(f"Found {val_failures} failures on val2017")

    print(f"\nFailed images saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

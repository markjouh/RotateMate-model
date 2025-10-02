import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import timm
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train rotation classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    return parser.parse_args()


def letterbox_resize(img, size):
    """Resize image with letterboxing to maintain aspect ratio."""
    c, h, w = img.shape
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    img = F.pad(img, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0)
    return img


def apply_color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
    """Apply color jitter using torch operations."""
    # Brightness
    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    img = img * brightness_factor

    # Contrast
    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
    mean = img.mean(dim=[1, 2], keepdim=True)
    img = (img - mean) * contrast_factor + mean

    # Saturation
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
    gray = img.mean(dim=0, keepdim=True)
    img = gray + (img - gray) * saturation_factor

    return img.clamp(0, 1)


class Dataset(TorchDataset):
    """Dataset for rotation classification."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, img_dir, img_size=224, augment=True):
        self.img_paths = [str(p) for p in Path(img_dir).glob("*.jpg")]
        self.img_size = img_size
        self.augment = augment
        self.normalize = transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx], ImageReadMode.RGB).float() / 255.0
        rotation = random.choice([0, 1, 2, 3])

        if rotation > 0:
            img = img.rot90(rotation, [1, 2])

        if self.augment:
            img = apply_color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1)
            # Add slight Gaussian noise
            img = img + torch.randn_like(img) * 0.02
            img = img.clamp(0, 1)

        img = letterbox_resize(img, self.img_size)
        img = self.normalize(img)
        return img, rotation


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total, total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total, total_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 224

    # Data
    train_dataset = Dataset("data/train2017", img_size=img_size, augment=True)
    val_dataset = Dataset("data/val2017", img_size=img_size, augment=False)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.workers, pin_memory=True)

    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")

    # Model
    model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=4)
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        train_acc, train_loss = train_epoch(model, tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"),
                                           criterion, optimizer, device)
        val_acc, val_loss = validate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {lr:.2e} *** BEST ***")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {lr:.2e} (patience: {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step()

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, "rotation_model.pth")
        print(f"Saved best model (val loss: {best_val_loss:.4f}): rotation_model.pth")
    else:
        print("Warning: No model was saved (training may have failed immediately)")


if __name__ == "__main__":
    main()

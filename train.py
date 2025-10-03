import argparse
import random
from pathlib import Path
from datetime import datetime

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


def resize_to_fit(img, size):
    """Resize image to fit within size x size, maintaining aspect ratio."""
    c, h, w = img.shape
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    return img


def pad_to_square(img, size):
    """Pad image to exactly size x size with black borders."""
    c, h, w = img.shape
    pad_h = size - h
    pad_w = size - w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    img = F.pad(img, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0)
    return img


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
        return len(self.img_paths) * 4  # Always 4 rotations per image

    def __getitem__(self, idx):
        # Each image has 4 variants (idx 0-3 is image 0 at 0°/90°/180°/270°)
        img_idx = idx // 4
        rotation = idx % 4

        img = read_image(self.img_paths[img_idx], ImageReadMode.RGB).float() / 255.0

        if rotation > 0:
            img = img.rot90(rotation, [1, 2])

        # Resize to fit (maintains aspect ratio)
        img = resize_to_fit(img, self.img_size)

        if self.augment:
            # Geometric augmentation: randomly apply hflip, vflip, or transpose
            aug_choice = random.random()
            if aug_choice < 0.25:
                # Horizontal flip
                img = torch.flip(img, dims=[2])
                rotation = [0, 3, 2, 1][rotation]
            elif aug_choice < 0.5:
                # Vertical flip
                img = torch.flip(img, dims=[1])
                rotation = [2, 1, 0, 3][rotation]
            elif aug_choice < 0.75:
                # Transpose (swap H and W)
                img = img.transpose(1, 2)
                rotation = [1, 2, 3, 0][rotation]
            # else: no geometric augmentation (25% of the time)

        # Pad to square
        img = pad_to_square(img, self.img_size)
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_model_state = None
    best_train_acc = 0.0
    best_val_acc = 0.0
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
            best_train_acc = train_acc
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {lr:.2e} *** BEST ***")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {lr:.2e} (patience: {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step()

    # Save best model with descriptive filename
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rotation_model_{timestamp}_train{best_train_acc:.1f}_val{best_val_acc:.1f}.pth"
        torch.save(best_model_state, filename)
        print(f"Saved best model: {filename}")

        # Also save as default name for backwards compatibility
        torch.save(best_model_state, "rotation_model.pth")
    else:
        print("Warning: No model was saved (training may have failed immediately)")


if __name__ == "__main__":
    main()

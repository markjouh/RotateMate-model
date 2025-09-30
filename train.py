import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import timm
import os
from pathlib import Path
from tqdm import tqdm
import random
import coremltools as ct
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
args = parser.parse_args()

# Config
BATCH_SIZE = 256
EPOCHS = args.epochs
LR = args.lr
PATIENCE = args.patience
IMG_SIZE = 224  # Match model's training resolution
NUM_WORKERS = 26

# Dataset
class RotationDataset(Dataset):
    def __init__(self, img_dir, augment=False):
        self.img_paths = [str(p) for p in Path(img_dir).glob("*.jpg")]
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Fast JPEG decode directly to tensor
        img = read_image(self.img_paths[idx], ImageReadMode.RGB).float() / 255.0
        rotation = random.choice([0, 1, 2, 3])  # 0°, 90°, 180°, 270°

        # Rotate with efficient tensor ops
        if rotation == 1:
            img = img.rot90(1, [1, 2])  # 90° CCW
        elif rotation == 2:
            img = img.rot90(2, [1, 2])  # 180°
        elif rotation == 3:
            img = img.rot90(3, [1, 2])  # 270° CCW

        # Letterbox resize
        img = letterbox_resize(img, IMG_SIZE)

        # Augmentation
        if self.augment:
            img = apply_augmentation(img)

        # Normalize
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, rotation

# Fast letterbox resize using PyTorch
def letterbox_resize(img, size):
    c, h, w = img.shape
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize with bilinear (fast, close to LANCZOS quality)
    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

    # Pad to square
    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    img = F.pad(img, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0)
    return img

def apply_augmentation(img):
    # Color jitter
    brightness = 1.0 + (random.random() * 0.2 - 0.1)  # 0.9-1.1
    contrast = 1.0 + (random.random() * 0.2 - 0.1)
    saturation = 1.0 + (random.random() * 0.2 - 0.1)

    img = img * brightness
    mean = img.mean(dim=0, keepdim=True)
    img = (img - mean) * contrast + mean

    # Saturation adjustment
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    img = gray.unsqueeze(0) + (img - gray.unsqueeze(0)) * saturation

    # Gaussian noise
    img = img + torch.randn_like(img) * 0.02

    return img.clamp(0, 1)

# Data
train_dataset = RotationDataset("data/train2017", augment=True)
val_dataset = RotationDataset("data/val2017", augment=False)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=4)
model = model.cuda()

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    val_loss /= len(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {optimizer.param_groups[0]['lr']:.2e} *** BEST ***")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}, LR {optimizer.param_groups[0]['lr']:.2e} (patience: {patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    scheduler.step()

# Load best model for export
model.load_state_dict(best_model_state)
torch.save(best_model_state, "rotation_model.pth")
print(f"Saved best model (val loss: {best_val_loss:.4f}): rotation_model.pth")

# Export to CoreML
model.eval().cpu()
example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
traced_model = torch.jit.trace(model, example_input)

# ImageNet normalization parameters for ImageType
# Converts: (pixel/255 - mean) / std
# To CoreML format: pixel * scale + bias
scale = 1 / (0.226 * 255.0)  # Using average std for scale
bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, IMG_SIZE, IMG_SIZE), scale=scale, bias=bias)],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS26,
)

mlmodel.save("RotationClassifier.mlpackage")

# INT8 quantization
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
mlmodel_compressed = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
mlmodel_compressed.save("RotationClassifier.mlpackage")
print("Exported: RotationClassifier.mlpackage (INT8 quantized)")
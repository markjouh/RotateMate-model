import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import random
import coremltools as ct

# Config
BATCH_SIZE = 256
EPOCHS = 10
LR = 3e-3
IMG_SIZE = 224  # Match model's training resolution
NUM_WORKERS = 26

# Dataset
class RotationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = list(Path(img_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        rotation = random.choice([0, 1, 2, 3])  # 0°, 90°, 180°, 270°

        # Rotate without resampling artifacts
        if rotation == 1:
            img = img.transpose(Image.ROTATE_270)
        elif rotation == 2:
            img = img.transpose(Image.ROTATE_180)
        elif rotation == 3:
            img = img.transpose(Image.ROTATE_90)

        if self.transform:
            img = self.transform(img)
        return img, rotation

# Transforms
def letterbox_transform(img):
    # Scale to fit exactly inside IMG_SIZE x IMG_SIZE (upscale or downscale), pad with black
    w, h = img.size
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    padded.paste(img, (paste_x, paste_y))
    return padded

transform = transforms.Compose([
    transforms.Lambda(letterbox_transform),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data
train_dataset = RotationDataset("data/train2017", transform)
val_dataset = RotationDataset("data/val2017", transform)
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
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%, LR {optimizer.param_groups[0]['lr']:.2e}")
    scheduler.step()

# Save
torch.save(model.state_dict(), "rotation_model.pth")
print("Saved: rotation_model.pth")

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
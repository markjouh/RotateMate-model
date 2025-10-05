#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class RotatedImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        rotation = random.randint(0, 3)

        if rotation > 0:
            image = image.rotate(-90 * rotation, expand=True)

        return self.transform(image), rotation


def letterbox_transform(img):
    # Scale to fit in 224x224 while preserving aspect ratio
    img.thumbnail((224, 224), Image.LANCZOS)
    # Create black 224x224 canvas
    canvas = Image.new('RGB', (224, 224), (0, 0, 0))
    # Center the image
    offset = ((224 - img.width) // 2, (224 - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def get_transform():
    """Get the transform pipeline used for training"""
    return transforms.Compose([
        transforms.Lambda(letterbox_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    # Create run directory with timestamp
    run_id = datetime.now().strftime("%m%d_%H%M")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")

    # Load dataset (255,502 images total)
    image_dir = Path("images")
    all_images = sorted(list(image_dir.glob("*.jpg")))

    # Fixed 10k validation set, rest for training
    val_images = all_images[:10000]
    train_images = all_images[10000:]

    # Transforms with letterboxing
    transform = get_transform()

    # Training configuration
    batch_size = 64
    epochs = 30
    lr = 3e-4

    train_data = RotatedImageDataset(train_images, transform)
    val_data = RotatedImageDataset(val_images, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, persistent_workers=True)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True, num_classes=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    final_confidences = []
    final_predictions = []
    final_labels = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_confidences = []
        epoch_predictions = []
        epoch_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                max_probs, preds = probs.max(1)
                epoch_confidences.extend(max_probs.cpu().numpy())
                epoch_predictions.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        val_losses.append(epoch_val_loss / len(val_loader))

        # Save best model and store final epoch data
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            final_confidences = epoch_confidences
            final_predictions = epoch_predictions
            final_labels = epoch_labels

        correct = sum(p == l for p, l in zip(epoch_predictions, epoch_labels))
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {100.*correct/len(epoch_labels):.2f}%')

    # Save best model
    model_path = run_dir / 'model.pth'
    torch.save(best_model_state, model_path)
    print(f"Best model saved to {model_path}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(run_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confidence threshold analysis (using best model's epoch)
    final_confidences = np.array(final_confidences)
    final_predictions = np.array(final_predictions)
    final_labels = np.array(final_labels)

    thresholds = np.arange(1.0, 0.24, -0.01)
    results = []

    for thresh in thresholds:
        confident_mask = final_confidences >= thresh
        if confident_mask.sum() > 0:
            confident_acc = (final_predictions[confident_mask] == final_labels[confident_mask]).mean() * 100
            unsure_pct = (1 - confident_mask.mean()) * 100
            results.append((thresh, confident_acc, unsure_pct))
        else:
            results.append((thresh, 0, 100))

    # Plot confidence analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    threshs, accs, unsures = zip(*results)

    ax1.plot(threshs, accs, marker='o', linewidth=2, markersize=3)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy on Confident Predictions (%)')
    ax1.set_title('Accuracy vs Confidence Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', alpha=0.1)

    ax2.plot(threshs, unsures, marker='o', linewidth=2, markersize=3, color='orange')
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Unsure Predictions (%)')
    ax2.set_title('Unsure Rate vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', alpha=0.1)

    plt.tight_layout()
    plt.savefig(run_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nConfidence Threshold Analysis:")
    print(f"{'Threshold':<12} {'Accuracy (%)':<15} {'Unsure (%)':<12}")
    print("-" * 40)
    for thresh, acc, unsure in results:
        print(f"{thresh:<12.2f} {acc:<15.2f} {unsure:<12.2f}")

    print(f"\nRun complete. Results saved to {run_dir}")

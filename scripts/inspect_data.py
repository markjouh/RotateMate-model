#!/usr/bin/env python3
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import from train.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from train import RotatedImageDataset, get_transform

def visualize_preprocessing(n=9):
    """Visualize n random preprocessed images"""
    # Load all images once (path relative to project root)
    image_dir = Path(__file__).parent.parent / "images"
    all_images = sorted(list(image_dir.glob("*.jpg")))
    transform = get_transform()

    # Create figure
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if n > 1 else [axes]

    def refresh_images(_event=None):
        """Refresh with new random images"""
        sample_images = random.sample(all_images, n)
        dataset = RotatedImageDataset(sample_images, transform)

        for idx in range(n):
            # Get sample from dataset (exactly as dataloader would)
            img_tensor, rotation = dataset[idx]

            # Denormalize for visualization
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_denorm = img_tensor.numpy() * std + mean
            img_denorm = np.clip(img_denorm, 0, 1)

            # Plot
            axes[idx].clear()
            axes[idx].imshow(img_denorm.transpose(1, 2, 0))
            axes[idx].set_title(f'Label: {rotation} ({rotation * 90}°)', fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        fig.canvas.draw()

    # Initial display
    refresh_images()

    # Add button and key binding
    fig.text(0.5, 0.02, 'Press R or click here to refresh', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.canvas.mpl_connect('key_press_event', lambda event: refresh_images() if event.key == 'r' else None)
    fig.canvas.mpl_connect('button_press_event', refresh_images)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    visualize_preprocessing(n)

import random

import torch
import torch.nn.functional as F


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


def apply_augmentation(img):
    """Apply color jitter and Gaussian noise augmentation."""
    brightness = 1.0 + (random.random() * 0.2 - 0.1)
    contrast = 1.0 + (random.random() * 0.2 - 0.1)
    saturation = 1.0 + (random.random() * 0.2 - 0.1)

    img = img * brightness
    mean = img.mean(dim=0, keepdim=True)
    img = (img - mean) * contrast + mean

    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    img = gray.unsqueeze(0) + (img - gray.unsqueeze(0)) * saturation
    img = img + torch.randn_like(img) * 0.02

    return img.clamp(0, 1)

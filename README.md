# RotateMate Model Training

Image rotation classification (0°, 90°, 180°, 270°) using MobileNetV4-small trained on COCO.

## Pipeline Overview

**1. Download** - COCO train/val/test splits (~20GB compressed)

**2. Preprocess** - GPU-accelerated preprocessing using Kornia
- Loads images in batches, applies rotation + resize on GPU
- Saves preprocessed tensors to `*_preprocessed/` directories
- Runtime: ~2-3 minutes on H100
- Eliminates CPU bottleneck during training

**3. Train** - MobileNetV4-small with 4-class rotation classification
- Uses preprocessed tensors for fast training
- 10 epochs with early stopping (patience=3)
- Saves best model to `checkpoints/best_model.pth`

**4. Export** - Converts to CoreML with INT8 quantization for iOS deployment
- Outputs: `exports/model.mlpackage` (~1.75MB)

## Setup

```bash
./setup.sh
source venv/bin/activate
```

## Usage

```bash
# GPU-accelerated preprocessing (run once, 2-3 min)
python -m src.preprocess

# Train single model
python train.py

# Train specific steps
python train.py --steps train export

# Hyperparameter sweep (trains N models, reports best)
python train.py --learning-rate 1e-4 2e-4 --weight-decay 1e-4 5e-5 --seeds 0 1 2
```

## Training Strategy

**Model:** MobileNetV4-small (ImageNet pretrained, 4-class head)

**Optimization:**
- AdamW optimizer (LR=1e-4, weight decay=1e-4)
- Linear warmup (1 epoch) → Cosine annealing
- Gradient clipping (max_norm=1.0)
- Label smoothing (0.1)

**Augmentation:**
- ColorJitter (brightness/contrast/saturation ±20%, hue ±5%, p=0.5)
- Gaussian noise (std=0.05, p=0.3)
- Applied on GPU via Kornia during training

**Training:** Batch size 1024, 10 epochs with early stopping (patience=3)

## Configuration

Edit `configs/h100.yaml` for hyperparameters:
- `batch_size: 1024`
- `learning_rate: 1e-4`
- `warmup_epochs: 1`
- `label_smoothing: 0.1`
- `quantization: "int8"` - Export format (options: "int8", "fp16", "none")

## Output

- `checkpoints/best_model.pth` - PyTorch checkpoint
- `exports/model.mlpackage` - CoreML INT8 quantized (~1.75MB)
- `logs/training_*.log` - Training logs with metrics

# RotateMate Model Training

Image rotation classification (0°, 90°, 180°, 270°) using MobileNetV4-small trained on COCO.

## Setup

```bash
./setup.sh
source venv/bin/activate
```

## Usage

```bash
# Train model (downloads data automatically)
python train.py

# Train specific steps
python train.py --steps train export

# Hyperparameter sweep
python train.py --learning-rate 1e-3 2e-3 --weight-decay 1e-4 5e-5 --seeds 0 1 2
```

## Pipeline

**1. Download** - COCO train/val splits (~20GB compressed, automatic on first run)

**2. Train** - MobileNetV4-small with 4-class rotation classification
- Loads JPEG images on-the-fly with standard PyTorch DataLoader
- Applies rotation (0°/90°/180°/270°) during loading
- 10 epochs with early stopping (patience=3)
- Saves best model to `checkpoints/best_model.pth`

**3. Export** - Converts to CoreML with INT8 quantization for iOS
- Outputs: `exports/model.mlpackage` (~1.75MB)

## Training Details

**Model:** MobileNetV4-small (ImageNet pretrained, 4-class head)

**Optimization:**
- AdamW optimizer (LR=1e-3, weight decay=1e-4)
- Cosine annealing learning rate schedule
- Early stopping (patience=3)

**Data:**
- 118k COCO train images × 4 rotations = 473k samples
- 5k COCO val images × 4 rotations = 20k samples
- Batch size: 2048 (optimized for H100 80GB)

## Configuration

Edit `configs/h100.yaml`:
- `batch_size: 2048` - Batch size for training
- `learning_rate: 1e-3` - Initial learning rate
- `epochs: 10` - Maximum epochs
- `patience: 3` - Early stopping patience
- `quantization: "int8"` - Export format (options: "int8", "fp16", "none")

## Output

- `checkpoints/best_model.pth` - PyTorch checkpoint
- `exports/model.mlpackage` - CoreML INT8 quantized (~1.75MB)
- `logs/training_*.log` - Training logs with metrics
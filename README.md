# RotateMate Model Training

Image rotation classification using MobileNetV4. Trained on COCO, optimized for H100.

## Setup

```bash
./setup.sh
source venv/bin/activate
```

## Usage

```bash
# Train a single model
python train.py

# Train with specific steps
python train.py --steps train export

# Hyperparameter sweep
python train.py --learning-rate 1e-4 2e-4 --weight-decay 1e-4 5e-5 --seeds 0 1 2
```

## Configuration

Edit `configs/h100.yaml`:
- MobileNetV4-small backbone (ImageNet pretrained)
- Batch size 1024 for H100 80GB
- Mixed precision (FP16/TF32)
- Cosine LR schedule with early stopping

## Output

- `checkpoints/best_model.pth` - PyTorch checkpoint
- `exports/model.mlpackage` - CoreML (FP16 quantized)
- `logs/training_*.log` - Training logs

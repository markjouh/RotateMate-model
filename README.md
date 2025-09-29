# RotateMate Model Training

Image rotation classification using MobileViT V2. Optimized for NVIDIA H100 (80GB HBM3).

## Setup

```bash
./setup.sh
source venv/bin/activate
```

## Usage

Run complete pipeline:
```bash
python run.py --steps all
```

Or run individual steps:
```bash
python run.py --steps download    # Download COCO dataset
python run.py --steps process     # Create rotated shards
python run.py --steps train       # Train model
python run.py --steps export      # Export to CoreML/ONNX
```

## Configuration

The `config.yaml` is optimized for H100 80GB:
- Batch size: 2048 (fits comfortably in 80GB)
- Workers: 20 data loading and processing (26 vCPUs)
- Mixed precision FP16 + TF32 enabled
- Full multiprocessing support on x86_64

## Output

- Model: `checkpoints/best_model.pth`
- CoreML: `exports/model.mlmodel`
- Logs: `logs/training_*.log`

## Performance

On H100: ~15-20 minutes for full COCO dataset (10 epochs)
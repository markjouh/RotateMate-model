# RotateMate Model Training

Image rotation classification using MobileViT V2. Optimized for NVIDIA GH200 (96GB HBM3).

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

The `config.yaml` is pre-optimized for GH200:
- Batch size: 1024 (96GB HBM3)
- Workers: 32 (72 ARM cores available)
- Mixed precision FP16 enabled

## Output

- Model: `checkpoints/best_model.pth`
- CoreML: `exports/model.mlmodel`
- Logs: `logs/training_*.log`

## Performance

On GH200: ~20 minutes for full COCO dataset (10 epochs)
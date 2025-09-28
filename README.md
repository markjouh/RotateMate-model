# RotateMate Model Training

Image rotation classification using MobileViT V2.

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

Edit `config.yaml` to adjust:
- `batch_size`: Based on GPU memory
- `epochs`: Training iterations
- `num_workers`: Data loading threads

## Output

- Model: `checkpoints/best_model.pth`
- CoreML: `exports/model.mlmodel`
- ONNX: `exports/model.onnx`
- Logs: `logs/training_*.log`
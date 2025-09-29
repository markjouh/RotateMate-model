# RotateMate Model Training

Image rotation classification using MobileViT V2. Optimized for Lambda Cloud H100 PCIe instances.

## Target Hardware

**Lambda Cloud H100 PCIe Instance:**
- GPU: NVIDIA H100 80GB HBM2e
- CPU: 26 vCPUs (x86_64)
- RAM: 200 GiB
- Storage: 1 TiB NVMe SSD
- Price: ~$2.49/hour

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
python run.py --steps download    # Download COCO dataset (~20GB)
python run.py --steps process     # Create rotated shards (~80GB)
python run.py --steps train       # Train model
python run.py --steps export      # Export to CoreML/ONNX
```

## Configuration

The `config.yaml` is optimized for H100 80GB:
- **Batch size**: 2048 (uses ~40GB of 80GB HBM2e)
- **Workers**: 20 for data loading (optimal for 26 vCPUs)
- **Cache**: 8 shards in RAM
- **Mixed precision**: FP16 with TF32 for matmuls
- **Storage**: ~100GB total disk usage

## Output

- Model: `checkpoints/best_model.pth`
- CoreML: `exports/model.mlmodel` (iOS deployment)
- Logs: `logs/training_*.log`
- Total disk usage: ~100GB (well within 1TB SSD)

## Performance

On Lambda Cloud H100:
- Training: ~15-20 minutes for full COCO dataset (10 epochs)
- Processing: ~10 minutes to create all shards
- Cost: ~$1.25 for complete training run
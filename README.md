# RotateMate Model Training

Image rotation classification using Google's MobileNetV4 (small). Optimized for Lambda Cloud H100 PCIe instances.

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
python run.py --steps train       # Train model
python run.py --steps export      # Export to CoreML/ONNX
```

### Parameter Sweep

Run automated learning-rate/weight-decay sweeps (defaults: LR ∈ {5e-5, 1e-4, 2e-4}, WD ∈ {5e-5, 1e-4}, seeds ∈ {0,1}):

```bash
python sweep.py --learning-rate 1e-4 3e-4 1e-3 --weight-decay 1e-4 5e-5 --seeds 0 1
```

Results and checkpoints are stored under `checkpoints/sweep_*`, and `summary.json` lists the top runs.

## Configuration

The `config.yaml` is optimized for H100 80GB:
- **Backbone**: `mobilenetv4_conv_small` from timm with ImageNet weights
- **Batch size**: 1024 (comfortably fits in 80GB HBM2e)
- **Workers**: 16 CPU workers keep the GPU fed
- **Preprocessing**: rotations applied on the fly with torchvision
- **Mixed precision**: FP16 with TF32 for Hopper matmuls
- **Storage**: ~45GB total disk usage (COCO + checkpoints)

## Output

- Model: `checkpoints/best_model.pth`
- CoreML: `exports/model.mlmodel` (iOS deployment)
- Logs: `logs/training_*.log`
- Total disk usage: ~45GB (well within 1TB SSD)

## Performance

On Lambda Cloud H100:
- Training: ~15 minutes for full COCO dataset (10 epochs)
- Cost: ~$1.25 for complete training run

# Lambda Cloud H100 Setup Guide

## Quick Start

1. **Launch Instance**
   ```bash
   # Lambda Cloud Console: Select H100 PCIe (1x)
   # Region: Any available (Virginia/Texas typically have best availability)
   # SSH into instance after launch
   ```

2. **Clone and Setup**
   ```bash
   git clone https://github.com/markjouh/RotateMate-model.git
   cd RotateMate-model
   ./setup.sh
   source venv/bin/activate
   ```

3. **Run Training**
   ```bash
   python run.py --steps all
   ```

## Instance Details

- **GPU**: NVIDIA H100 80GB HBM2e (PCIe version)
- **vCPUs**: 26 (Intel/AMD x86_64)
- **RAM**: 200 GiB DDR5
- **Storage**: 1 TiB NVMe SSD
- **Network**: 100 Gbps
- **CUDA**: Pre-installed (12.1+)
- **Cost**: ~$2.49/hour

## Storage Layout

```
/home/ubuntu/
├── RotateMate-model/
│   ├── data/
│   │   ├── raw/         # ~20GB (downloaded zips)
│   │   ├── coco/        # ~20GB (extracted images)
│   │   └── shards/      # ~80GB (processed tensors)
│   ├── checkpoints/     # ~200MB (model checkpoints)
│   └── logs/           # <100MB
```

Total: ~120GB used of 1TB available

## Performance Expectations

- **Download**: 2-3 minutes (100 Gbps network)
- **Processing**: 8-10 minutes (20 CPU workers)
- **Training**: 15-20 minutes (10 epochs)
- **Total time**: ~30 minutes
- **Total cost**: ~$1.25

## Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Training progress
tail -f logs/training_*.log

# System resources
htop
```

## Tips

1. **Preemptible Instances**: Lambda offers spot-like pricing at ~$1.99/hour
2. **Persistent Storage**: Attach a persistent filesystem if needed
3. **Multi-GPU**: Scale to 8x H100 for larger models
4. **Batch Size**: Can increase to 4096 if needed (uses ~60GB)

## Troubleshooting

- **CUDA OOM**: Reduce batch_size in config.yaml
- **CPU bottleneck**: Reduce num_workers to 16
- **Slow download**: Use wget with multiple connections
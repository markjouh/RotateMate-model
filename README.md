# RotateMate Model

Image rotation classification for iOS deployment.

## Task
Classify image rotation into 4 classes: 0°, 90°, 180°, 270°

## Model
- **Architecture**: MobileNetV4-small
- **Pretrained**: ImageNet
- **Head**: 4-class classifier (transfer learning)
- **Deployment**: iOS 26 via CoreML with INT8 quantization

## Data
- **Training**: COCO train2017 (118k images)
- **Validation**: COCO val2017 (5k images)
- Rotations generated on-the-fly during training

## Hardware
- H100 80GB GPU
- 26 vCPUs

## Setup
```bash
./setup.sh
```

## Training
```bash
python train.py
```

## Output
- `rotation_model.pth` - PyTorch checkpoint
- `RotationClassifier.mlpackage` - INT8 quantized CoreML model for iOS
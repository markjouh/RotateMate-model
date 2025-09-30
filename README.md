# RotateMate Model

Efficient and accurate classification model for the RotateMate iOS app.

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
- Random rotation, color jitter, Gaussian noise for each image on each epoch

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
- `RotationClassifier.mlpackage` - Exported CoreML model
## Project Context

RotateMate is a mobile app that helps the user correct mis-rotated images in their photo gallery using an on-device image classification model. The model can act as a filter to narrow down the set of images for the user to review. It may also work in the background autonomously, rotating images to the correct orientation automatically when confidence is high.

## Model

- Base: Pretrained `mobilenetv4_conv_small.e2400_r224_in1k1` from `timm`
- Output: 4 classes `[0, 1, 2, 3]`, each representing the number of clockwise 90° rotations needed to upright the image
- Input resolution: 224×224 (training and inference)
- Export format: CoreML `.mlpackage` with 8-bit quantization

## Training Hardware

The model will be trained on an Apple M4 Max with 16 CPU cores, 40 GPU cores, and 64GB of unified memory.

Due to the relatively constrained compute, keeping training fast and efficient is a priority. Training runs should complete in under 1 hour.

## Dataset

We will be using images from the [AVA dataset](https://huggingface.co/datasets/Iceclear/AVA), a collection of 255,502 photos from photography competitions. Compared to ImageNet datasets like COCO, it has the advantages of not being limited to photos with a definite subject from a fixed set of classes, and lower odds of being mis-rotated due to greater curation.

Training uses on-the-fly augmentation to generate (data, label) samples from each image by applying random rotations (0°, 90°, 180°, 270°). This allows us to generate up to 4 distinct training samples from each source image.

## Training Goals

- Target accuracy: 97%+
- Minimize confident misclassifications - the app can leverage user review when the model is uncertain
- Training time: under 1 hour per run

## Implementation

Keep the codebase as simple as possible.

- Keep all the code related to training in a single file, `train.py`.
- Always implement things in a straightforward and concise way.
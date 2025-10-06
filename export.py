#!/usr/bin/env python3
import torch
import timm
import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig
import coremltools.optimize.coreml.experimental as cto_experimental
import argparse
from pathlib import Path
from PIL import Image
import random
from train import get_transform


def get_calibration_data(num_samples=128):
    """Load calibration images from the dataset"""
    image_dir = Path("images")
    all_images = sorted(list(image_dir.glob("*.jpg")))

    # Use validation set images (first 10k)
    val_images = all_images[:10000]
    sample_images = random.sample(val_images, min(num_samples, len(val_images)))

    transform = get_transform()

    calibration_data = []
    for img_path in sample_images:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)  # Add batch dimension
        calibration_data.append({"input": tensor.numpy()})

    return calibration_data


def export_to_coreml(model_path, output_path='RotateMate.mlpackage'):
    # Load model
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to('cpu').eval()

    # Trace model
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML with iOS 17+ target for better quantization support
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
        minimum_deployment_target=ct.target.iOS17
    )

    # Apply 8-bit weight quantization
    print("Applying 8-bit weight quantization...")
    weight_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode='linear_symmetric'))
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, weight_config)

    # Apply 8-bit activation quantization
    print("Applying 8-bit activation quantization with calibration data...")
    calibration_data = get_calibration_data(num_samples=128)
    activation_config = OptimizationConfig(
        global_config=cto_experimental.OpActivationLinearQuantizerConfig(mode='linear_symmetric')
    )
    mlmodel = cto_experimental.linear_quantize_activations(mlmodel, activation_config, calibration_data)
    print(f"Used {len(calibration_data)} calibration samples")

    mlmodel.save(output_path)
    print(f"Exported to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to CoreML')
    parser.add_argument('--model-path', required=True, help='Path to the PyTorch model (.pth file)')
    parser.add_argument('--output-path', default='RotateMate.mlpackage', help='Output CoreML package path')
    args = parser.parse_args()
    export_to_coreml(args.model_path, args.output_path)

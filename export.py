import argparse

import torch
import timm
import coremltools as ct


def parse_args():
    parser = argparse.ArgumentParser(description='Export PyTorch model to CoreML')
    parser.add_argument('--checkpoint', type=str, default='rotation_model.pth', help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='RotationClassifier.mlpackage', help='Output CoreML package path')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')
    parser.add_argument('--ios-version', type=str, default='iOS26', help='Minimum iOS deployment target (e.g., iOS26, iOS17)')
    return parser.parse_args()


def export_to_coreml(checkpoint_path, output_path, img_size=224, quantize=True, ios_target='iOS26'):
    """Export PyTorch model to CoreML with optional INT8 quantization."""

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval().cpu()

    # Trace model
    print(f"Tracing model with input size {img_size}x{img_size}...")
    example_input = torch.randn(1, 3, img_size, img_size)
    traced_model = torch.jit.trace(model, example_input)

    # ImageNet normalization for CoreML (per-channel)
    # Training uses: (pixel/255 - mean) / std
    # CoreML applies: pixel * scale + bias
    # Therefore: scale = 1/(255*std), bias = -mean/std
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    scale = [1.0 / (255.0 * s) for s in IMAGENET_STD]
    bias = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

    # Get iOS target
    ios_target_map = {
        'iOS26': ct.target.iOS26,
        'iOS18': ct.target.iOS18,
        'iOS17': ct.target.iOS17,
        'iOS16': ct.target.iOS16,
    }
    target = ios_target_map.get(ios_target, ct.target.iOS26)

    # Convert to CoreML
    print(f"Converting to CoreML (target: {ios_target})...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, img_size, img_size), scale=scale, bias=bias)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=target,
    )

    if quantize:
        print("Applying INT8 quantization...")
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)

    # Save
    mlmodel.save(output_path)
    print(f"Saved CoreML model to: {output_path}")

    # Print model info
    if quantize:
        print("Quantization: INT8")
    else:
        print("Quantization: None (FP32)")


def main():
    args = parse_args()
    export_to_coreml(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.img_size,
        quantize=args.quantize,
        ios_target=args.ios_version
    )


if __name__ == "__main__":
    main()

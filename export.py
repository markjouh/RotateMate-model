import torch
import coremltools as ct


def export_to_coreml(model, img_size, output_path="RotationClassifier.mlpackage"):
    """Export model to CoreML with INT8 quantization."""
    model.eval().cpu()
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

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, img_size, img_size), scale=scale, bias=bias)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS26,
    )

    mlmodel.save(output_path)

    # INT8 quantization
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_compressed = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    mlmodel_compressed.save(output_path)

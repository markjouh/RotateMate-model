"""Model export utilities for RotateMate."""

import logging
from pathlib import Path

import torch
from timm import create_model

logger = logging.getLogger(__name__)


def _quantize_model(mlmodel, quantization_mode="int8"):
    """Quantize CoreML model."""
    try:
        import coremltools as ct

        if quantization_mode == "int8":
            logger.info("Quantizing to INT8...")
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)

        elif quantization_mode == "fp16":
            logger.info("Quantizing to FP16...")
            from coremltools.models.neural_network import quantization_utils
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

        return mlmodel

    except Exception as err:
        logger.warning("Quantization failed: %s", err)
        return mlmodel


def _load_model(checkpoint_path, model_config):
    """Load trained model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = create_model(
        model_config["name"],
        pretrained=False,
        num_classes=model_config["num_classes"]
    )

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def export_model(checkpoint_path, export_config, model_config, output_dir, image_size=256):
    """Export model to CoreML and/or ONNX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint_path, model_config)
    exported = {}

    # CoreML export
    if export_config.get("coreml", {}).get("enabled", False):
        try:
            import coremltools as ct

            logger.info("Exporting to CoreML...")

            example_input = torch.zeros(1, 3, image_size, image_size)
            traced_model = torch.jit.trace(model, example_input)

            class_labels = [str(i) for i in range(model_config["num_classes"])]

            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="image", shape=example_input.shape)],
                classifier_config=ct.ClassifierConfig(class_labels),
                compute_units=ct.ComputeUnit.ALL,
            )

            quantization = export_config["coreml"].get("quantization", "int8")
            if quantization and quantization != "none":
                mlmodel = _quantize_model(mlmodel, quantization)

            mlmodel_path = output_dir / "model.mlpackage"
            mlmodel.save(str(mlmodel_path))
            exported["coreml"] = mlmodel_path
            logger.info("Saved CoreML: %s", mlmodel_path)

        except ImportError:
            logger.warning("coremltools not installed")
        except Exception as err:
            logger.error("CoreML export failed: %s", err)

    # ONNX export
    if export_config.get("onnx", {}).get("enabled", False):
        try:
            logger.info("Exporting to ONNX...")
            example_input = torch.zeros(1, 3, image_size, image_size)
            onnx_path = output_dir / "model.onnx"

            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                opset_version=export_config["onnx"].get("opset_version", 17),
                input_names=["input"],
                output_names=["output"],
            )

            exported["onnx"] = onnx_path
            logger.info("Saved ONNX: %s", onnx_path)

        except Exception as err:
            logger.error("ONNX export failed: %s", err)

    return exported
"""Model export utilities for RotateMate."""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from timm import create_model
from timm.data import resolve_model_data_config

from .trainer import ModelWrapper

logger = logging.getLogger(__name__)


def _convert_to_fp16(mlmodel):
    """Convert CoreML model to FP16 using official quantization API."""
    try:
        from coremltools.models.neural_network import quantization_utils
        logger.info("Converting model to FP16...")
        return quantization_utils.quantize_weights(mlmodel, nbits=16)
    except Exception as err:
        logger.warning("FP16 conversion failed: %s. Keeping FP32.", err)
        return mlmodel


def _load_model(checkpoint_path, model_config):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Verify checkpoint structure
    required_keys = ['model_state_dict', 'val_acc']
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(f"Invalid checkpoint: missing keys {missing}")

    # Recreate model architecture
    model_core = create_model(
        model_config["name"],
        pretrained=False,
        num_classes=model_config["num_classes"],
        drop_rate=model_config.get("drop_rate", 0.0),
        drop_path_rate=model_config.get("drop_path_rate", 0.0),
    )

    # Get normalization stats
    data_cfg = resolve_model_data_config(model_core)
    mean = model_config.get("normalize", {}).get("mean", data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = model_config.get("normalize", {}).get("std", data_cfg.get("std", (0.229, 0.224, 0.225)))

    # Load weights
    model = ModelWrapper(model_core, mean, std)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["val_acc"]


def export_model(checkpoint_path, export_config, model_config, output_dir, image_size=256):
    """Export model to CoreML and/or ONNX formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, val_acc = _load_model(checkpoint_path, model_config)
    exported = {}

    # CoreML export
    if export_config.get("coreml", {}).get("enabled", False):
        try:
            import coremltools as ct

            logger.info("Exporting to CoreML...")

            # Step 1: Trace model
            example_input = torch.zeros(1, 3, image_size, image_size)
            traced_model = torch.jit.trace(model, example_input)
            logger.info("Model traced successfully")

            # Step 2: Convert to CoreML
            class_labels = [str(i) for i in range(model_config["num_classes"])]

            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="image", shape=example_input.shape)],
                classifier_config=ct.ClassifierConfig(class_labels),
                compute_units=ct.ComputeUnit.ALL,
            )
            logger.info("Model converted to CoreML successfully")

            # Step 3: Optional FP16 quantization
            if export_config["coreml"].get("fp16", True):
                mlmodel = _convert_to_fp16(mlmodel)

            # Step 4: Save as mlpackage (modern format)
            mlmodel_path = output_dir / "model.mlpackage"
            mlmodel.save(str(mlmodel_path))
            exported["coreml"] = mlmodel_path
            logger.info("Saved CoreML model: %s", mlmodel_path)

        except ImportError:
            logger.warning("coremltools not installed - skipping CoreML export")
        except Exception as err:
            logger.error("CoreML export failed: %s", err, exc_info=True)

    # ONNX export
    if export_config.get("onnx", {}).get("enabled", False):
        try:
            logger.info("Exporting ONNX...")
            example_input = torch.zeros(1, 3, image_size, image_size)
            onnx_path = output_dir / "model.onnx"

            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                opset_version=export_config["onnx"].get("opset_version", 17),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )

            exported["onnx"] = onnx_path
            logger.info("Exported ONNX: %s", onnx_path)

        except Exception as err:
            logger.error("ONNX export failed: %s", err)

    # Save export metadata
    metadata = {
        "val_acc": float(val_acc),
        "exported_formats": list(exported.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return exported

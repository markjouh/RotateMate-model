"""Model export utilities for RotateMate."""

import importlib
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from timm import create_model
from timm.data import resolve_model_data_config

from trainer import ModelWrapper

logger = logging.getLogger(__name__)


def _convert_coreml_model_to_fp16(mlmodel):
    """Attempt FP16 conversion across supported coremltools APIs."""
    try:
        from coremltools.models import utils as ct_utils
        convert_fn = getattr(ct_utils, "convert_neural_network_weights_to_fp16", None)
        if convert_fn is not None:
            return convert_fn(mlmodel)
    except Exception as err:
        logger.warning("FP16 conversion via coremltools.models.utils failed: %s", err)

    for module_path in (
        "coremltools.optimize.coreml.quantization_utils",
        "coremltools.optimize.coreml",
        "coremltools.models.neural_network.quantization_utils",
    ):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue

        quantize_fn = getattr(module, "quantize_weights", None)
        if quantize_fn is None:
            continue

        try:
            return quantize_fn(mlmodel, nbits=16)
        except Exception as err:
            logger.warning(
                "FP16 conversion via %s.quantize_weights failed: %s",
                module_path,
                err,
            )

    logger.warning("FP16 conversion not available; exporting Core ML model with float32 weights.")
    return mlmodel


def export_model(checkpoint_path, export_config, model_config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]

    model_core = create_model(
        model_config["name"],
        pretrained=model_config.get("pretrained", False),
        num_classes=model_config["num_classes"],
        drop_rate=model_config.get("drop_rate", 0.0),
        drop_path_rate=model_config.get("drop_path_rate", 0.0),
    )
    data_cfg = resolve_model_data_config(model_core)
    mean = model_config.get("normalize", {}).get("mean", data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = model_config.get("normalize", {}).get("std", data_cfg.get("std", (0.229, 0.224, 0.225)))

    model = ModelWrapper(model_core, mean, std)
    model.load_state_dict(model_state)
    model.eval()

    if export_config.get("coreml", {}).get("enabled", False):
        try:
            import coremltools as ct

            logger.info("Exporting Core ML...")
            example_input = torch.randn(1, 3, 256, 256)
            traced = torch.jit.trace(model, example_input)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                classifier_config=ct.ClassifierConfig(
                    class_labels=[str(i) for i in range(model_config["num_classes"])]
                ),
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS17,
            )

            if export_config["coreml"].get("fp16", True):
                mlmodel = _convert_coreml_model_to_fp16(mlmodel)

            mlmodel_path = output_dir / "model.mlmodel"
            mlmodel.save(str(mlmodel_path))
            exported["coreml"] = mlmodel_path
            logger.info("Exported: %s", mlmodel_path)

        except ImportError:
            logger.warning("coremltools not installed")
        except Exception as err:
            logger.error("Core ML export failed: %s", err)

    if export_config.get("onnx", {}).get("enabled", False):
        try:
            logger.info("Exporting ONNX...")
            example_input = torch.randn(1, 3, 256, 256)
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
            logger.info("Exported: %s", onnx_path)

        except Exception as err:
            logger.error("ONNX export failed: %s", err)

    metadata_path = output_dir / "export_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json_payload = {
            "val_acc": float(checkpoint["val_acc"]),
            "exported_formats": list(exported.keys()),
            "timestamp": datetime.now().isoformat(),
        }
        json.dump(json_payload, handle, indent=2)

    return exported

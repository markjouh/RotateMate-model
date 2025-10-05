#!/usr/bin/env python3
import torch
import timm
import coremltools as ct
import sys

def export_to_coreml(model_path='rotatemate_model.pth', output_path='RotateMate.mlpackage'):
    # Load model
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model = model.to('cpu').eval()

    # Trace model
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
        minimum_deployment_target=ct.target.iOS15
    )

    # Apply 8-bit weight quantization
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, nbits=8)
    mlmodel.save(output_path)
    print(f"Exported to {output_path}")

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'rotatemate_model.pth'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'RotateMate.mlpackage'
    export_to_coreml(model_path, output_path)

"""ONNX Model Exporter and Validation.

Exports PyTorch models to ONNX and numerically verifies ONNX Runtime outputs against PyTorch.
"""

from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from .tiny_cnn import TinyCNN
from .tiny_transformer import TinyTransformer


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    input_names: Tuple[str, ...] = ("input_0",),
    output_names: Tuple[str, ...] = ("logits",),
) -> str:
    """Exports a PyTorch model to ONNX format and verifies numerical equivalence.

    Args:
        model: PyTorch module in eval mode.
        sample_input: PyTorch sample tensor matching model input shape.
        output_path: Filepath where the .onnx file will be saved.
        input_names: Input tensor name(s).
        output_names: Output tensor name(s).

    Returns:
        output_path on success.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. PyTorch reference forward
    model.eval()
    with torch.no_grad():
        torch_out = model(sample_input).detach().cpu().numpy()

    # 2. Export using standard PyTorch ONNX exporter
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=18,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes={input_names[0]: {0: "batch_size"}},
    )

    # 3. Verify ONNX Runtime output matches PyTorch
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(output_path, session_options, providers=["CPUExecutionProvider"])

    ort_inputs = {session.get_inputs()[0].name: sample_input.detach().cpu().numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    assert np.allclose(torch_out, ort_out, atol=1e-4, rtol=1e-3), (
        f"ONNX export mismatch: Max absolute difference = {max_diff:.6f}"
    )

    return output_path

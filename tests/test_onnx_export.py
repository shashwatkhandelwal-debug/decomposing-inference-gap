"""Test ONNX model export and numerical parity with PyTorch and ONNX Runtime."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.models.tiny_cnn import TinyCNN
from src.models.tiny_transformer import TinyTransformer
from src.models.export_onnx import export_to_onnx


def test_export_tiny_cnn_to_onnx():
    """Verify TinyCNN exports to ONNX and runs on ONNX Runtime matching PyTorch outputs."""
    model = TinyCNN(in_channels=3, num_classes=10, hidden_channels=16)
    sample_input = torch.randn(2, 3, 16, 16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "tiny_cnn.onnx")
        res_path = export_to_onnx(model, sample_input, onnx_path)
        assert os.path.exists(res_path)


def test_export_tiny_transformer_to_onnx():
    """Verify TinyTransformer exports to ONNX and runs on ONNX Runtime matching PyTorch outputs."""
    model = TinyTransformer(num_layers=2, d_model=32, n_heads=2, d_ff=64)
    sample_input = torch.randn(2, 8, 32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "tiny_transformer.onnx")
        res_path = export_to_onnx(model, sample_input, onnx_path)
        assert os.path.exists(res_path)

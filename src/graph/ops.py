"""Pure NumPy implementations and definitions for supported IR operators.

Includes shape inference and standard forward implementations.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


def op_matmul(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Matrix multiplication: A @ B."""
    return np.matmul(inputs[0], inputs[1])


def op_add(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Elementwise addition with broadcasting."""
    return np.add(inputs[0], inputs[1])


def op_relu(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(inputs[0], 0)


def op_gelu(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Gaussian Error Linear Unit approximation."""
    x = inputs[0]
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def op_conv2d(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """2D Convolution: inputs=[X, W, (optional) B].

    Shape convention:
      X: (N, C_in, H, W)
      W: (C_out, C_in, kH, kW)
      B: (C_out,) [optional]
    Attributes:
      stride: Tuple[int, int] (default: (1, 1))
      padding: Tuple[int, int] (default: (0, 0))
    """
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 else None

    stride = attributes.get("stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)
    padding = attributes.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    sH, sW = stride
    pH, pW = padding

    out_H = (H + 2 * pH - kH) // sH + 1
    out_W = (W + 2 * pW - kW) // sW + 1

    # Pad input if needed
    if pH > 0 or pW > 0:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (pH, pH), (pW, pW)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_pad = x

    # Vectorized im2col implementation for robust performance
    col = np.zeros((N, C_in, kH, kW, out_H, out_W), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            col[:, :, i, j, :, :] = x_pad[:, :, i : i + sH * out_H : sH, j : j + sW * out_W : sW]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_H * out_W, -1)
    w_row = w.reshape(C_out, -1).T
    out = np.matmul(col, w_row)

    if b is not None:
        out += b.reshape(1, -1)

    out = out.reshape(N, out_H, out_W, C_out).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(out)


def op_batchnorm2d(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Inference-mode BatchNorm2d: inputs=[X, scale, bias, mean, var].

    Formula: y = (x - mean) / sqrt(var + eps) * scale + bias
    """
    x = inputs[0]
    scale = inputs[1].reshape(1, -1, 1, 1)
    bias = inputs[2].reshape(1, -1, 1, 1)
    mean = inputs[3].reshape(1, -1, 1, 1)
    var = inputs[4].reshape(1, -1, 1, 1)
    eps = attributes.get("eps", 1e-5)

    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * scale + bias


def op_layernorm(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """LayerNorm across last dimension: inputs=[X, gamma, beta]."""
    x = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    eps = attributes.get("eps", 1e-5)
    axis = attributes.get("axis", -1)

    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta


def op_softmax(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Softmax over specified axis (default: -1) with optional scaling and numerical stability."""
    x = inputs[0]
    scale = attributes.get("scale", 1.0)
    if scale != 1.0:
        x = x * scale
    axis = attributes.get("axis", -1)
    shift_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def op_reducemean(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Computes mean along specified axis."""
    x = inputs[0]
    axis = attributes.get("axis", 1)
    if isinstance(axis, list):
        axis = tuple(axis)
    keepdims = attributes.get("keepdims", False)
    return np.mean(x, axis=axis, keepdims=keepdims)


def op_reshape(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Reshape tensor to target shape specified in attributes or second input."""
    x = inputs[0]
    shape = attributes.get("shape")
    if shape is None and len(inputs) > 1:
        shape = tuple(inputs[1].tolist())
    return np.reshape(x, shape)


def op_transpose(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Transpose / permute axes."""
    x = inputs[0]
    axes = attributes.get("axes", None)
    return np.transpose(x, axes)


def op_flatten(inputs: List[np.ndarray], attributes: Dict[str, Any]) -> np.ndarray:
    """Flatten tensor from start_dim to end_dim."""
    x = inputs[0]
    start_dim = attributes.get("start_dim", 1)
    end_dim = attributes.get("end_dim", -1)
    shape = x.shape
    if end_dim == -1:
        end_dim = len(shape) - 1

    pre_dim = int(np.prod(shape[:start_dim])) if start_dim > 0 else 1
    flat_dim = int(np.prod(shape[start_dim : end_dim + 1]))
    post_dim = int(np.prod(shape[end_dim + 1 :])) if end_dim + 1 < len(shape) else 1

    new_shape = []
    if start_dim > 0:
        new_shape.extend(shape[:start_dim])
    new_shape.append(flat_dim)
    if end_dim + 1 < len(shape):
        new_shape.extend(shape[end_dim + 1 :])

    return np.reshape(x, tuple(new_shape))


OP_REGISTRY: Dict[str, Callable[[List[np.ndarray], Dict[str, Any]], np.ndarray]] = {
    "MatMul": op_matmul,
    "Add": op_add,
    "ReLU": op_relu,
    "GELU": op_gelu,
    "Conv2d": op_conv2d,
    "BatchNorm2d": op_batchnorm2d,
    "LayerNorm": op_layernorm,
    "Softmax": op_softmax,
    "Reshape": op_reshape,
    "Transpose": op_transpose,
    "Flatten": op_flatten,
    "ReduceMean": op_reducemean,
}

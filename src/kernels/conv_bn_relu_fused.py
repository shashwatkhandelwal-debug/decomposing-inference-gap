"""Hand-written Fused Conv2d + BatchNorm2d + ReLU Kernel.

Folds batch normalization parameters directly into convolutional weights and biases,
then computes convolution with fused ReLU activation without intermediate buffers.

Algebraic BatchNorm Folding:
  Given Conv output: y_conv = W * x + b
  BatchNorm: y_bn = gamma * (y_conv - mu) / sqrt(var + eps) + beta
  Expanding:
    y_bn = [gamma / sqrt(var + eps)] * (W * x + b - mu) + beta
         = [gamma * W / sqrt(var + eps)] * x + [gamma * (b - mu) / sqrt(var + eps) + beta]
  Therefore:
    W_fused = W * [gamma / sqrt(var + eps)].reshape(C_out, 1, 1, 1)
    b_fused = gamma * (b - mu) / sqrt(var + eps) + beta
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from ..graph.ops import op_conv2d


def fold_batchnorm_into_conv(
    conv_w: np.ndarray,
    conv_b: Optional[np.ndarray],
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    bn_eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Algebraically folds BatchNorm parameters into Conv2d weight and bias.

    Args:
        conv_w: Conv2d weights of shape (C_out, C_in, kH, kW)
        conv_b: Optional Conv2d bias of shape (C_out,)
        bn_gamma: BatchNorm scale parameter of shape (C_out,)
        bn_beta: BatchNorm shift parameter of shape (C_out,)
        bn_mean: BatchNorm running mean of shape (C_out,)
        bn_var: BatchNorm running variance of shape (C_out,)
        bn_eps: Numerical stability epsilon

    Returns:
        Tuple of (W_fused, b_fused)
    """
    scale = bn_gamma / np.sqrt(bn_var + bn_eps)
    w_fused = conv_w * scale.reshape(-1, 1, 1, 1)

    if conv_b is not None:
        b_fused = scale * (conv_b - bn_mean) + bn_beta
    else:
        b_fused = -scale * bn_mean + bn_beta

    return w_fused.astype(conv_w.dtype), b_fused.astype(conv_w.dtype)


def conv_bn_relu_fused(
    x: np.ndarray,
    w_fused: np.ndarray,
    b_fused: np.ndarray,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    apply_relu: bool = True,
) -> np.ndarray:
    """Computes Conv2d with folded parameters and in-place ReLU activation."""
    out = op_conv2d([x, w_fused, b_fused], {"stride": stride, "padding": padding})
    if apply_relu:
        np.maximum(out, 0, out=out)
    return out

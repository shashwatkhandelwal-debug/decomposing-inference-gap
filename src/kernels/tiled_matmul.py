"""Hand-written Cache-Blocked / Tiled Matrix Multiplication.

Decomposes large matrix multiplications into small tiles (block_size x block_size)
to maximize L1/L2 cache residency and minimize cache misses.
"""

from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True, parallel=False)
    def _tiled_matmul_numba(A: np.ndarray, B: np.ndarray, C: np.ndarray, block_size: int) -> None:
        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[1]

        for i0 in range(0, M, block_size):
            i_max = min(i0 + block_size, M)
            for k0 in range(0, K, block_size):
                k_max = min(k0 + block_size, K)
                for j0 in range(0, N, block_size):
                    j_max = min(j0 + block_size, N)

                    # Micro-kernel within the cache-resident block
                    for i in range(i0, i_max):
                        for k in range(k0, k_max):
                            a_ik = A[i, k]
                            for j in range(j0, j_max):
                                C[i, j] += a_ik * B[k, j]


def tiled_matmul(A: np.ndarray, B: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Computes A @ B using cache blocking (tiling).

    Args:
        A: 2D array of shape (M, K)
        B: 2D array of shape (K, N)
        block_size: Tile dimension (e.g. 16, 32, 64, 128)

    Returns:
        Result array C of shape (M, N)
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]

    A_contig = np.ascontiguousarray(A, dtype=np.float32)
    B_contig = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A_contig.shape
    N = B_contig.shape[1]
    C = np.zeros((M, N), dtype=np.float32)

    if NUMBA_AVAILABLE:
        _tiled_matmul_numba(A_contig, B_contig, C, block_size)
    else:
        # Vectorized tiled multiplication across block columns
        for i0 in range(0, M, block_size):
            i_max = min(i0 + block_size, M)
            for k0 in range(0, K, block_size):
                k_max = min(k0 + block_size, K)
                for j0 in range(0, N, block_size):
                    j_max = min(j0 + block_size, N)
                    # Sub-tile matmul using sliced NumPy operations
                    C[i0:i_max, j0:j_max] += np.matmul(
                        A_contig[i0:i_max, k0:k_max],
                        B_contig[k0:k_max, j0:j_max],
                    )

    return C

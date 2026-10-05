# Project Specification: Decomposing the Inference Speedup Gap

**A Research Implementation of a Minimal ML Inference Engine**

> Central Thesis: Production inference runtimes (ONNX Runtime, TensorRT) outperform naive framework execution (PyTorch eager mode) by a large, often-quoted-but-rarely-decomposed margin. This project builds a minimal graph executor from scratch, implements the three dominant optimization techniques — operator fusion, memory layout optimization, and quantization — as independent, toggleable stages, and measures how much of the real-world speedup gap each one actually closes, in isolation and combined, against real ONNX Runtime and (if hardware allows) TensorRT baselines. The goal is not to build a faster runtime. The goal is to produce a defensible, quantitative attribution of *where the speedup comes from*.

---

## 0. Non-Negotiable Constraints

1. **No black-box optimization.** Every optimization technique (fusion, layout, quantization) must be implemented by hand in Python/NumPy/PyTorch low-level ops — not by calling `torch.compile()`, `torch.jit`, or any existing graph compiler. The point is to understand *why* the optimization works, not to invoke a tool that already does it.
2. **Ground truth must be independently reproducible.** Every reported number must come from a script that can be re-run and will reproduce the number within stated variance (report mean ± std over ≥100 trials for all timing measurements).
3. **Correctness before speed.** Every optimized execution path must be numerically validated against the naive baseline (max absolute error and relative error reported) before its timing is reported. A fast-but-wrong result is a bug, not a finding.
4. **Negative results are mandatory, not optional.** If a "real" optimization is implemented and does *not* produce a speedup (this is likely and expected for at least one stage — see Phase 4), that must be reported prominently, with a hypothesis for why, not buried or omitted.
5. **All benchmarking must control for warmup, caching, and system noise.** Minimum 20 warmup iterations, minimum 100 trials, explicit outlier trimming, CPU thread count pinned, environment fingerprinted.
6. **The owner must be able to explain every profiler trace produced.** No number should be reported unless it has been traced to its source (a specific kernel, a specific memory access pattern, a specific cache-line behavior).

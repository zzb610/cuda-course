# Project 05: PyTorch Custom Op Benchmark

Recommended after: Module 17  
Estimated time: 20-35 小时

## Goal

把一个简单但完整的 CUDA kernel 做成 PyTorch custom op。这个项目不是追求复杂算法，而是训练现代 CUDA 工程闭环：schema、launcher、CUDA stream、correctness、fake/meta、benchmark、packaging 和文档。

## Suggested Operator

任选一个：

- `scale_add(x, y, alpha)`
- `fused_bias_gelu(x, bias)`
- `rmsnorm_forward(x, weight, eps)`
- `quantize_per_token(x)`

优先选择你能写出清楚 PyTorch reference 的算子。

## Required Deliverables

- Operator schema。
- C++ launcher，包含 dtype/device/shape/contiguity checks。
- CUDA implementation。
- Python wrapper。
- Fake/meta 或 shape-only implementation 设计。
- Correctness tests，覆盖 edge cases 和 dtype-specific tolerance。
- Benchmark against PyTorch eager baseline。
- Nsight Systems 或 Nsight Compute observation。
- Final integration memo。

## Final Memo Template

```markdown
Operator:
Schema:
Reference:
Supported dtypes:
Supported shapes:
Unsupported cases:
CUDA stream behavior:
Correctness tolerance:
Benchmark method:
Profiler observation:
PyTorch compile/fake tensor notes:
How this would enter vLLM/SGLang:
Fallback:
```

## Expert Bar

你通过本项目不是因为 kernel 快，而是因为另一个工程师能安装、调用、测试、测量，并清楚知道它什么时候该用、什么时候不该用。

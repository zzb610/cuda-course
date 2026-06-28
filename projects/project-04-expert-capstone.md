# Project 04: Expert Capstone

Recommended after: Module 11  
Estimated time: 20-40 小时

## Goal

完成一个能体现专家判断的 CUDA 项目。重点是完整工程闭环，不是单一技巧。

## Candidate Topics

- Fused elementwise + reduction operator。
- Tiled GEMM 或 convolution progression。
- Attention-like simplified kernel。
- Multi-stream image/data pipeline。
- Histogram 或 irregular workload。
- Small CUDA library with Python or C++ interface。

## Required Deliverables

- Problem statement and constraints。
- CPU or library baseline。
- CUDA baseline。
- Optimized CUDA implementation。
- Tests, including edge cases。
- Benchmark harness。
- Profiling report。
- Source grounding and version notes。
- Architecture caveats。
- Final engineering memo。

## Final Engineering Memo

Use this structure:

```markdown
Problem:
Constraints:
Hardware/toolchain:
Baseline:
Bottleneck evidence:
Optimization attempts:
Final result:
Correctness evidence:
Portability caveats:
What I would try next:
```

## Rubric

- Correctness and tests: 30%
- Measurement and profiling: 20%
- Optimization reasoning: 20%
- Code quality: 15%
- Explanation and caveats: 15%

## Expert Bar

You pass the expert bar if another engineer can reproduce your build, run your tests, understand your bottleneck evidence, and decide whether your optimization is worth maintaining.

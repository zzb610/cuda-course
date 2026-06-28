# Project 02: Memory-Optimized Kernels

Recommended after: Module 05  
Estimated time: 10-14 小时

## Goal

围绕 transpose 或 reduction，做一次有证据的内存优化。

## Choose One Track

Track A: Matrix transpose

- naive transpose。
- tiled transpose。
- 支持非方阵和边界 tile。
- 用 Nsight Compute 对比内存相关指标。

Track B: Reduction

- atomic baseline。
- shared-memory block reduction。
- CUB comparison。
- 支持任意长度。

## Deliverables

- Baseline and optimized code。
- Correctness tests。
- Nsight Compute notes。
- Profiling report。
- 简短结论：优化为什么有效，什么时候可能无效。

## Rubric

- Correctness across edge cases: 30%
- Memory reasoning: 25%
- Profiling evidence: 25%
- Code quality and explanation: 20%

## Reject Conditions

- 只有 speedup，没有 correctness test。
- 没有输入规模和硬件信息。
- 同时改太多因素，无法解释原因。

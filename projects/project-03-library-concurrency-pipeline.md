# Project 03: Library And Concurrency Pipeline

Recommended after: Module 08  
Estimated time: 10-16 小时

## Goal

构建一个结合 CUDA 库和 streams 的 pipeline，并用 Nsight Systems 验证并发。

## Suggested Problem

数据分块处理：

1. H2D copy chunk。
2. library operation，例如 CUB reduce 或 cuBLAS GEMM。
3. custom postprocess kernel。
4. D2H copy result。

## Requirements

- 至少两个 streams。
- 使用 pinned memory 做 overlap 实验。
- 有单 stream baseline。
- 用 Nsight Systems 证明 timeline 是否 overlap。
- 清楚说明库调用使用的 stream。

## Deliverables

- Source code。
- Timeline evidence。
- Timing table。
- Explanation of stream dependencies。
- Caveat: 哪些硬件或输入条件会影响结果。

## Rubric

- Correct stream semantics: 25%
- Library API use: 20%
- Profiling evidence: 25%
- Correctness and tests: 15%
- Explanation: 15%

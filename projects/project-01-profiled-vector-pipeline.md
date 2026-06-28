# Project 01: Profiled Vector Pipeline

Recommended after: Module 02  
Estimated time: 6-10 小时

## Goal

把一个简单 vector operation 做成完整 CUDA 小项目：正确性、错误检查、计时、报告都齐全。

## Requirements

- 支持 `y[i] = a * x[i] + y[i]`。
- 支持至少 5 个输入规模，包括 0、1、非 block 整除长度、大数组。
- 使用 CUDA error-checking helper。
- 用 CUDA events 分别测量 H2D、kernel、D2H。
- 用 CPU baseline 验证结果。

## Deliverables

- Source code。
- Build/run instructions。
- Test results。
- Profiling report v1。
- 1 页反思：目前 bottleneck 是什么？下一步会优化什么？

## Rubric

- Correctness: 40%
- Error handling: 20%
- Measurement quality: 20%
- Explanation: 20%

## Stretch

- 支持多种 block size 参数。
- 比较 pageable 和 pinned memory，但不要过度解读，先记录。

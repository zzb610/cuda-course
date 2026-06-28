# Plan: Module 19 Triton/TileLang/CUDA/CUTLASS 精品讲义扩写

> 历史写作计划草稿，不作为课程事实来源。当前 Triton、TileLang、CUDA/CUTLASS、DeepGEMM 等事实以 `resources.md`、对应 `modules/` 正文和当前官方文档/源码为准。

## 阶段划分

### Stage 1: 研究与大纲设计（Orchestrator 完成）
- 读取现有文件，保留精华（手工车床比喻、对比总览表、心智模型、四层对照、评估清单）
- 设计 5 层次结构大纲
- 确定 6 段代码的内容和结构
- 确定 4 个 Mermaid 图的内容

### Stage 2: 核心内容分块写作（并行子代理）
- **Writer_A**: 写 1. 问题背景 + 2. 直觉类比 + 3. 硬件机制（Triton 编译流程、TileLang 架构）
- **Writer_B**: 写 4. 代码路径 — Triton 详解（6段代码中的 Triton 部分）
- **Writer_C**: 写 5. 代码路径 — TileLang 详解 + CUDA C++ 对比
- **Writer_D**: 写 6. 真实系统落点（PyTorch 2.0、OpenAI、DeepSeek、vLLM/SGLang）+ DSL 代价分析

### Stage 3: 整合与润色（Orchestrator 完成）
- 合并所有子代理输出
- 插入 Mermaid 图
- 添加学习目标、mental model、lab、练习阶梯、checkpoint、常见错误、extension
- 确保总长度 > 25000 字符
- 质量检查（URL 来源、中文讲解、英文术语保留）
- 写回文件

## 文件传播
- `/Users/bowenyuchi/Documents/cuda-course/modules/module-19-triton-tilelang-cuda-cutlass-comparison.md` — 最终输出文件

## 技能加载
- 无特定技能需要（直接写作任务）
- 使用 `general-writing` 作为内容质量标准参考

## 质量标准
- 最终文件 > 25000 字符
- 保留原文件所有精华内容
- 6 段精品代码 + 逐行注释
- 4 个 Mermaid 图
- 5 个 Lab
- 完整学习目标 / 练习阶梯 / checkpoint
- 资料来源包含 URL

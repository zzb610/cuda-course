# CUDA 编程从小白到专家

现代 GPU Kernel、CUDA 工程实践与 LLM 推理系统精品课。

![CUDA course knowledge map](diagrams/exported/course-knowledge-map.png)

这是一套面向 CUDA 初学者到专家级工程实践的系统课程。课程主线不是 API 罗列，而是从“能写正确 kernel”推进到“能测量、优化、工程化、接入 PyTorch/vLLM/SGLang，并理解现代 GPU 硬件与推理系统”。

## 快速入口

| 入口 | 用途 |
|---|---|
| [课程总览](OUTLINE.md) | 课程目标、20 周结构、学习方法 |
| [写作与维护规范](WRITING_GUIDE.md) | 讲义风格、代码要求、图示导出规则 |
| [资料来源](resources.md) | NVIDIA、PyTorch、vLLM、Triton、TileLang 等来源索引 |
| [GPU 架构地图](GPU_ARCHITECTURE_MAP.md) | Volta、Turing、Ampere、Hopper、Blackwell 的软件/硬件对应关系 |
| [硬件心智模型](HARDWARE_MENTAL_MODELS.md) | SM、warp、memory coalescing、Tensor Core、NVLink/RDMA 等速查 |
| [诊断测评](assessments/diagnostic.md) | 开始学习前检查 C/C++、命令行、GPU 与 CUDA 基础 |

## 推荐阅读顺序

1. 先读 [课程总览](OUTLINE.md)，明确最终能力和 20 周路线。
2. 做 [诊断测评](assessments/diagnostic.md)，确认是否需要补 C/C++、Linux 命令行或编译基础。
3. 按 `Module 00 -> Module 19` 顺序阅读讲义。
4. 每 3 到 4 个模块完成一个项目，并用 [评分标准](assessments/rubric.md) 做自查。
5. 每次优化都写 measurement note：硬件、输入规模、编译参数、测量工具、结果、结论。
6. 高级阶段反复对照 [GPU 架构地图](GPU_ARCHITECTURE_MAP.md) 和 [资料来源](resources.md)，避免把某代 GPU 的结论误用到所有硬件。

## 课程目录

| 模块 | 主题 | 讲义 | 主图 |
|---|---|---|---|
| 00 | 环境、工具链与 host/device 心智模型 | [Module 00](modules/module-00-environment-and-prerequisites.md) | [PNG](diagrams/exported/module-00-cuda-host-device-model.png) |
| 01 | 执行模型与第一个 kernel | [Module 01](modules/module-01-execution-model-first-kernel.md) | [PNG](diagrams/exported/module-01-cuda-execution-hierarchy.png) |
| 02 | GPU 内存管理、数据搬运与正确性 | [Module 02](modules/module-02-memory-and-correctness.md) | [PNG](diagrams/exported/module-02-cuda-memory-spaces.png) |
| 03 | 内存层级、coalescing 与 tiling | [Module 03](modules/module-03-memory-hierarchy-coalescing-tiling.md) | [PNG](diagrams/exported/module-03-gpu-memory-hierarchy-pyramid.png) |
| 04 | 同步、atomics 与 reduction | [Module 04](modules/module-04-synchronization-atomics-reduction.md) | [PNG](diagrams/exported/module-04-reduction-tree.png) |
| 05 | Profiling 与性能基础 | [Module 05](modules/module-05-profiling-performance-foundations.md) | [PNG](diagrams/exported/module-05-gpu-profiling-tools.png) |
| 06 | 并行算法模式 | [Module 06](modules/module-06-parallel-algorithm-patterns.md) | [PNG](diagrams/exported/module-06-parallel-algorithm-patterns.png) |
| 07 | Thrust、CUB、cuBLAS 与 CUDA 库生态 | [Module 07](modules/module-07-libraries-thrust-cub-cublas.md) | [PNG](diagrams/exported/module-07-cuda-libraries-hierarchy.png) |
| 08 | Streams、events、overlap 与 CUDA Graphs | [Module 08](modules/module-08-streams-events-concurrency-graphs.md) | [PNG](diagrams/exported/module-08-cuda-stream-overlap.png) |
| 09 | 架构感知优化 | [Module 09](modules/module-09-architecture-aware-optimization.md) | [PNG](diagrams/exported/module-09-gpu-architecture-occupancy.png) |
| 10 | 生产级 CUDA 工程 | [Module 10](modules/module-10-production-cuda-engineering.md) | [PNG](diagrams/exported/module-10-cuda-production-engineering.png) |
| 11 | Expert Capstone Studio | [Module 11](modules/module-11-expert-capstone-studio.md) | [PNG](diagrams/exported/module-11-capstone-workflow.png) |
| 12 | GPU 架构演进：Volta 到 Blackwell | [Module 12](modules/module-12-gpu-architecture-generations.md) | [PNG](diagrams/exported/module-12-gpu-architecture-matrix.png) |
| 13 | Tensor Core、WMMA、MMA 与 GEMM 分层 | [Module 13](modules/module-13-tensor-cores-wmma-mma-gemm.md) | [PNG](diagrams/exported/module-13-gemm-tensor-core.png) |
| 14 | PTX、SASS、WGMMA、TMA 与异步流水线 | [Module 14](modules/module-14-ptx-sass-wgmma-tma-pipeline.md) | [PNG](diagrams/exported/module-14-async-pipeline-ptx.png) |
| 15 | CUTLASS、CuTe、DeepGEMM 与现代 GEMM 工程 | [Module 15](modules/module-15-cutlass-cute-deepgemm-modern-gemm-engineering.md) | [PNG](diagrams/exported/module-15-cutlass-deepgemm.png) |
| 16 | NCCL、NVLink、GPUDirect RDMA 与 DeepEP | [Module 16](modules/module-16-nccl-nvlink-gpudirect-rdma-deepep.md) | [PNG](diagrams/exported/module-16-nccl-communication-topology.png) |
| 17 | PyTorch Custom Ops 到 vLLM/SGLang 注册路径 | [Module 17](modules/module-17-pytorch-vllm-sglang-custom-op-integration.md) | [PNG](diagrams/exported/module-17-pytorch-custom-op.png) |
| 18 | Attention、KV Cache、PagedAttention、MLA 与推理算子 | [Module 18](modules/module-18-attention-kv-cache-pagedattention-mla-inference-kernels.md) | [PNG](diagrams/exported/module-18-pagedattention-kv-cache.png) |
| 19 | Triton、TileLang 与 CUDA/CUTLASS 对比 | [Module 19](modules/module-19-triton-tilelang-cuda-cutlass-comparison.md) | [PNG](diagrams/exported/module-19-cuda-triton-tilelang-comparison.png) |

## 项目

| 项目 | 主题 |
|---|---|
| [Project 01](projects/project-01-profiled-vector-pipeline.md) | 从 vector add 到可测量 pipeline |
| [Project 02](projects/project-02-memory-optimized-kernels.md) | transpose/reduction 的内存优化与 profiling |
| [Project 03](projects/project-03-library-concurrency-pipeline.md) | CUB/cuBLAS 与 streams overlap |
| [Project 04](projects/project-04-expert-capstone.md) | 专家级最终项目 |
| [Project 05](projects/project-05-pytorch-custom-op-benchmark.md) | CUDA kernel 到 PyTorch custom op |
| [Project 06](projects/project-06-modern-gemm-fp8-deepgemm-study.md) | FP8/FP4 GEMM、CUTLASS/DeepGEMM 与 vLLM |
| [Project 07](projects/project-07-moe-communication-deepep-rdma.md) | MoE dispatch/combine、NCCL、RDMA、DeepEP |
| [Project 08](projects/project-08-triton-tilelang-cuda-comparison.md) | 同一算子在 CUDA/Triton/TileLang/CUTLASS 下的对比 |

## 评估与检查点

- [Diagnostic](assessments/diagnostic.md)：开课前诊断。
- [Weekly Checkpoints](assessments/weekly-checkpoints.md)：每周 checkpoint。
- [Rubric](assessments/rubric.md)：项目与 capstone 评分标准。

## 图示

所有可编辑源图在 [diagrams/](diagrams/)，所有 GitHub 可直接预览的 PNG 在 [diagrams/exported/](diagrams/exported/)。

图示使用 Excalidraw 手写风。修改图后按以下顺序检查和导出：

```bash
python3 scripts/check_diagram_fonts.py
node scripts/export_excalidraw_with_excalifont.mjs
```

## 代码起步

`starter-code/` 提供最小 CUDA CMake 工程：

```bash
cmake -S starter-code -B build
cmake --build build
```

实际编译需要本机已安装 NVIDIA driver、CUDA Toolkit 和支持 CUDA 的编译器。版本兼容性以本机 `nvidia-smi`、`nvcc --version` 与 NVIDIA 官方文档为准。

## 仓库结构

```text
.
├── modules/          # 20 个课程模块
├── projects/         # 8 个项目 brief
├── assessments/      # 诊断、周检查点、评分标准
├── diagrams/         # Excalidraw 源图与导出 PNG
├── starter-code/     # 最小 CUDA CMake 示例
├── assets/fonts/     # Excalifont 与中文手写 fallback 字体
├── scripts/          # 图示字体检查与导出脚本
├── OUTLINE.md        # 课程总览
└── resources.md      # 资料来源
```

## 阅读建议

每个模块都很长，建议不要只在 GitHub 页面快速滑过。推荐做法是：先读模块顶部图示和学习目标，再读故事线和类比，然后直接跑代码实验，最后回到硬件机制和 profiler 章节做复盘。

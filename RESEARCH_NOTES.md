# Research Notes For The 精品 CUDA Course

访问日期：2026-06-28

这份笔记记录本轮课程重构使用的一手资料和课程设计结论。它不替代每课讲义，但用于约束后续写作不要脱离事实。

## NVIDIA Architecture And ISA

- CUDA Toolkit Release Notes 当前页面标题为 CUDA Toolkit 13.3。
- CUDA Samples README 标注支持 CUDA Toolkit 13.3。
- Ampere Tuning Guide 明确强调 Ampere 的 async global-to-shared copy、split arrive/wait barrier、第三代 Tensor Cores、TF32/BF16、A100 更大的 shared memory/L1/texture 组合容量。
- Hopper Tuning Guide 明确引入 Tensor Memory Accelerator，TMA 可搬运 1D 到 5D tensor，在 global memory 和 shared memory 之间异步搬运，并支持 thread block clusters 与 distributed shared memory。
- Blackwell Tuning Guide 延续 CUDA 编程模型并引入 Blackwell tuning 视角，包含 thread block clusters、distributed shared memory、第五代 NVLink 等主题。
- PTX ISA 9.3 文档包含 `mma.sync`、`wgmma.mma_async`、`cp.async.bulk.tensor`、`mbarrier`、`tcgen05` 等现代 kernel 必须理解的底层入口。

## Communication

- NVIDIA NCCL 官方资料把 NCCL 定位为面向 NVIDIA GPU 和 networking 优化的多 GPU、多节点通信 primitives，常见路径包括 all-reduce、all-gather、reduce-scatter、broadcast、send/receive 等。
- GPUDirect RDMA 文档把 GPU memory 暴露给第三方 PCIe peer device，例如 NIC、storage adapter。关键概念包括 PCI BAR、pin/unpin GPU memory、registration cache、nvidia-peermem、IOMMU caveat。
- DeepEP 的 README 将现代 MoE 通信落到现实：EP dispatch/combine、NVLink/RDMA、NCCL Gin backend、JIT、少占 SM 的通信路径、InfiniBand/RoCE caveat。当前 V2 明确说明 0-SM RDMA low-latency EP 不再支持；0-SM Engram/PP/CP 属于其他实验性 primitive，不能和 EP low-latency path 混同。

## PyTorch And Inference Frameworks

- PyTorch 官方 custom op 教程推荐用 `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` 或 stable ABI 方式注册 C++/CUDA op，并用 `torch.library.opcheck` 检查注册正确性。CUDA 实现应接入当前 CUDA stream。
- vLLM CustomOp 文档说明通过继承 `CustomOp`、使用 `@CustomOp.register("op_name")`、实现 `forward_xxx()` 接入框架，也支持 OOT device plugins 通过 `register_oot` 替换特定 op；本地源码中 OOT 默认按 in-tree 类名匹配，也可显式传 `name`。
- vLLM PagedAttention 文档说明 attention kernel 要兼容 paged KV cache，key/value cache 分块存储，并且这里的 block 概念不同于 GPU thread block。
- 本地 vLLM 源码中存在 `_custom_ops.py`、`csrc/torch_bindings.cpp`、CUTLASS/FP8/FP4/MoE/CMake 架构选择等真实案例。课程应引用这些路径作为阅读任务。

## Modern Kernel Libraries And DSLs

- DeepGEMM 是 DeepSeek 官方高性能 Tensor Core kernel library，覆盖 FP8/FP4/BF16 GEMM、MoE、JIT、SM90/SM100、TMA-aligned layout、PTX/SASS dump 等；vLLM vendored DeepGEMM 集成里出现 SM120/SM12x 编译与 scale-format 分支，课程中必须区分 DeepGEMM 上游 README 和框架集成支持矩阵。
- Triton 官方 matmul tutorial 用 program instance、block pointers、`tl.dot`、autotune configs 解释 tile-level matmul。
- TileLang 文档/README 包含 CUDA backend、Tensor Core policy、PTX MMA、`cp.async`、H100 Auto TMA/WGMMA、FlashMLA/DeepSeek MLA 等面向现代 kernel 的 DSL 能力；Blackwell/`tcgen05` 相关能力要以当前 release notes、README 和生成代码为准。

## Course Design Implications

1. 课程必须从“CUDA C++ 基础”升级到“现代 GPU kernel 工程”。
2. Tensor Core 不能只讲 WMMA，要沿着 WMMA -> MMA/PTX -> WGMMA -> TMA pipeline -> CUTLASS/DeepGEMM 讲。
3. 通信不能只讲 NCCL API，要讲 NVLink、GPUDirect RDMA、registration cache、MoE dispatch/combine、DeepEP。
4. 现实落点必须围绕 PyTorch custom op、vLLM/SGLang 接入、精度对齐、benchmark、profiler、CI。
5. 新编程模型必须和 CUDA/CUTLASS 对比，讲 Triton/TileLang 的抽象优势和边界。

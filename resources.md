# CUDA 课程资料源

访问日期：2026-06-28

## 当前版本假设

NVIDIA 官方 CUDA Toolkit Release Notes 页面标题显示当前文档为 CUDA Toolkit 13.3。本课程以 CUDA 13.3 文档作为当前参考，但不会把所有练习绑定到 13.3 独有特性。学习时请记录：

```bash
nvcc --version
nvidia-smi
cmake --version
```

## 官方主资料

- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUDA Toolkit Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- CUDA Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/index.html
- CUDA Samples: https://github.com/NVIDIA/cuda-samples
- Compute Sanitizer: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- Nsight Systems: https://docs.nvidia.com/nsight-systems/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Ampere Tuning Guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- GPUDirect RDMA: https://docs.nvidia.com/cuda/gpudirect-rdma/index.html

## 官方库资料

- CCCL source repository: https://github.com/NVIDIA/cccl
- CUB API documentation: https://nvidia.github.io/cccl/unstable/cub/api/
- Thrust API documentation: https://nvidia.github.io/cccl/unstable/thrust/api/
- cuBLAS: https://docs.nvidia.com/cuda/cublas/
- cuDNN: https://docs.nvidia.com/deeplearning/cudnn/
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- CUTLASS: https://github.com/NVIDIA/cutlass

## 现代 AI 推理与 Kernel 工程资料

- PyTorch Custom C++ and CUDA Operators: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- vLLM CustomOp design: https://docs.vllm.ai/en/stable/design/custom_op/
- vLLM PagedAttention design: https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM local source case study: `/Users/bowenyuchi/Documents/vllm_dev/vllm`
- SGLang docs: https://docs.sglang.ai/
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- DeepEP: https://github.com/deepseek-ai/DeepEP
- Triton docs: https://triton-lang.org/main/
- Triton matrix multiplication tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- TileLang docs: https://tilelang.com/
- TileLang matmul tutorial: https://tilelang.com/deeplearning_operators/matmul.html

## 工程资料

- CMake CUDA architectures property: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
- CMake `CMAKE_CUDA_ARCHITECTURES`: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html

## 使用原则

- API、工具、安装、GPU 架构能力、compute capability 表格都以官方当前文档为准。
- 社区博客只能作为辅助讲解，不能替代官方文档。
- 性能实验必须在本机 GPU 上重跑，不引用别人机器的 speedup 作为结论。
- 如果 CUDA Toolkit、driver、Nsight 工具版本不同，先记录差异，再调整实验。

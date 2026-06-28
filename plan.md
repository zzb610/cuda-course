# Plan: Module 17 扩写

> 历史写作计划草稿，不作为课程事实来源。当前 API、版本和框架行为以 `resources.md`、对应 `modules/` 正文以及本地源码核查结果为准。

## 阶段 1: 规划与资料整合
- 已读取现有 module-17-pytorch-vllm-sglang-custom-op-integration.md
- 已搜索补充资料：PyTorch custom op API、vLLM CustomOp、SGLang RadixAttention、torch.compile、fake tensor、dispatch 机制

## 阶段 2: 内容架构设计（5 层次结构）

### 层次 1: 问题背景
- 为什么需要 custom op 集成
- 从 kernel 到 framework 的 gap
- 现代推理框架的需求

### 层次 2: 直觉类比
- 发动机装车比喻（保留并深化）
- 分层心智模型

### 层次 3: 硬件机制
- CUDA stream、device、Tensor Core
- 多精度计算路径

### 层次 4: 代码路径
- PyTorch custom op 注册（TORCH_LIBRARY / torch.library.custom_op）
- PyTorch Extension 构建（JIT / setup.py / CMake / pyproject.toml）
- vLLM CustomOp 注册
- SGLang backend 集成
- pybind11 绑定
- autograd backward
- fake/meta implementation
- torch.compile 兼容
- torch.export 交互
- torch.library.opcheck

### 层次 5: 真实系统落点
- 推理框架中的调度
- 多 GPU 分发
- 性能 benchmark 设计
- 精度验证体系
- 打包和发布

## 阶段 3: 必须包含的元素清单

### Mermaid 图（至少 4 个）
1. custom op 分层架构图（保留原图，扩展）
2. PyTorch dispatch 链图
3. vLLM CustomOp 注册路径图
4. 精度验证流程图

### 精品代码（至少 5 段）
1. PyTorch custom op 完整注册骨架（支持多 dtype、多 device）
2. PyTorch fake/meta implementation 完整示例
3. torch.utils.cpp_extension JIT 编译示例
4. vLLM CustomOp 完整注册示例（继承、register、forward）
5. pybind11 Python-C++ 绑定示例

### 新增/深化内容
- TORCH_LIBRARY schema 语法详解
- TORCH_LIBRARY_IMPL dispatch 机制详解
- torch.library.custom_op（当前官方教程按 PyTorch 2.4+ 工作流讲解；具体 API 以安装版本为准）
- autograd backward 完整示例
- torch.compile 兼容性 + fake implementation
- torch.library.opcheck
- torch.export 和 custom op 交互
- torch.utils.cpp_extension JIT 编译（load_inline、load）
- setup.py 完整配置（CUDA 扩展）
- CMake 集成（find_package(Torch)）
- Python wheel 打包和发布
- setuptools + pyproject.toml 现代配置
- vLLM CustomOp 基类、注册、forward_xxx、OOT plugin
- vLLM CUDA graph 兼容性
- vLLM platform/backend 抽象
- vLLM model_config 与 op 交互
- SGLang RadixAttention + PagedAttention 集成
- SGLang custom backend 机制
- SGLang 与 vLLM 差异
- 精度验证：FP32/FP16/BF16/FP8 误差分析
- torch.testing.assert_close tolerance 设置
- 数值稳定性测试
- 性能 benchmark 完整设计
- PyTorch inductor + custom op 融合
- torch.compile 对 custom op 的优化
- 推理框架 op 调度（prefill vs decode）
- 多 GPU 环境下 custom op 分发

## 阶段 4: 质量检查清单
- [ ] 文件长度 > 25000 字符
- [ ] 完整学习目标、mental model、lab（>=4）、练习阶梯、checkpoint
- [ ] 常见错误、extension
- [ ] 代码可直接编译运行
- [ ] 资料来源包含 URL
- [ ] 中文讲解，保留英文术语
- [ ] 5 层次结构清晰
- [ ] 4+ Mermaid 图
- [ ] 5+ 精品代码段

## 阶段 5: 执行
- 使用 Python 脚本生成完整 Markdown 内容
- 写入 /Users/bowenyuchi/Documents/cuda-course/modules/module-17-pytorch-vllm-sglang-custom-op-integration.md

import re

def humanize_module17(text):
    changes = []
    
    # 删除记忆钩子
    old = '记忆钩子：**kernel 是零件，operator 是产品接口，framework integration 是生产系统。**'
    new = ""
    if old in text:
        text = text.replace(old, new)
        changes.append("删除记忆钩子")
    
    # 填充短语
    fillers = [
        ("值得注意的是，", ""),
        ("值得注意的是", ""),
        ("此外，", ""),
        ("此外", ""),
        ("值得一提的是", ""),
        ("需要指出的是", ""),
    ]
    
    for old_p, new_p in fillers:
        if old_p in text:
            text = text.replace(old_p, new_p)
            changes.append(f"删除填充短语: {old_p}")
    
    replacements = [
        ('很多 CUDA 课程停在"写一个 `.cu` 文件，跑出正确结果"。真实 AI 工程不会停在这里。现在写 CUDA 算子，通常要注册到 PyTorch，和 eager mode、`torch.compile`、fake tensor、autograd、dtype dispatch、device dispatch、benchmark、CI、wheel packaging 打交道。进入 vLLM / SGLang 这类推理框架后，还要考虑 custom op 的生命周期、CUDA graph、模型配置、KV cache、调度器和 fallback path。',
         '很多 CUDA 课程停在"写一个 `.cu` 文件，跑出正确结果"。真实 AI 工程不会停在这里。现在写 CUDA 算子，通常要注册到 PyTorch，和 eager mode、`torch.compile`、fake tensor、autograd、dtype dispatch、device dispatch、benchmark、CI、wheel packaging 打交道。进入 vLLM / SGLang 这类推理框架后，还要考虑 custom op 生命周期、CUDA graph、模型配置、KV cache、调度器和 fallback path。'),
        
        ('这节课把你从"会写 kernel 的人"推到"能交付框架算子的人"。同一个 CUDA kernel，只有被框架正确调用、测试、测量、打包、降级，才算工程完成。',
         '这节课把你从"会写 kernel"推到"能交付框架算子"。同一个 CUDA kernel，只有被框架正确调用、测试、测量、打包、降级，才算工程完成。'),
        
        ('单独跑通 kernel，就像发动机在台架上能转。PyTorch custom op 是把发动机装进车，让油门、仪表盘、刹车、冷却都能工作。vLLM / SGLang 集成则是把车放进车队调度系统，要求它在高峰期、不同路线、不同司机手里都稳定。',
         '单独跑通 kernel，像发动机在台架上能转。PyTorch custom op 是把发动机装进车，让油门、仪表盘、刹车、冷却都能工作。vLLM / SGLang 集成则是把车放进车队调度系统，要求它在高峰期、不同路线、不同司机手里都稳定。'),
        
        ('记忆钩子：**kernel 是零件，operator 是产品接口，framework integration 是生产系统。**',
         ''),
        
        ('每一层都有自己的失败模式。kernel 正确不代表 launcher 正确；launcher 正确不代表 PyTorch dispatch 正确；PyTorch op 正确不代表能被 vLLM scheduler 安全调用。一个 production-ready 的 custom op 必须打通所有层。',
         '每一层都有自己的失败模式。kernel 正确不代表 launcher 正确；launcher 正确不代表 PyTorch dispatch 正确；PyTorch op 正确不代表能被 vLLM scheduler 安全调用。production-ready 的 custom op 必须打通所有层。'),
        
        ('PyTorch 提供两条互补的注册路径，分别适用于不同场景：',
         'PyTorch 提供两条互补的注册路径：'),
        
        ('`TORCH_LIBRARY` 定义 operator schema，它是 PyTorch dispatcher 的**契约接口**。Schema 规定了：op 名字、参数类型、返回值、是否就地修改 (mutates)。',
         '`TORCH_LIBRARY` 定义 operator schema，它是 PyTorch dispatcher 的契约接口。Schema 规定 op 名字、参数类型、返回值、是否就地修改 (mutates)。'),
        
        ('> **核心心智模型**：Dispatcher 是"电话交换机"。Schema 是"电话号码簿"，`TORCH_LIBRARY_IMPL` 是"接线员把某个号码接到具体办公室"。你只需告诉交换机你的号码，来电会自动转到正确的办公室。',
         '> Dispatcher 像电话交换机。Schema 是电话号码簿，`TORCH_LIBRARY_IMPL` 是接线员把某个号码接到具体办公室。告诉交换机你的号码，来电自动转到正确办公室。'),
        
        ('当前 PyTorch 官方 custom Python operator 教程按 PyTorch 2.4+ 工作流讲解 `torch.library.custom_op`，允许**纯 Python** 注册 custom op，无需写 C++ 扩展文件。它适合快速原型和 Python-centric workflow。',
         '当前 PyTorch 官方 custom Python operator 教程按 PyTorch 2.4+ 工作流讲解 `torch.library.custom_op`，允许纯 Python 注册 custom op，无需写 C++ 扩展文件。适合快速原型和 Python-centric workflow。'),
        
        ('课程里不要鼓励学生复制这段当完整实现，而是让学生理解每个检查为什么存在。',
         '课程里不要鼓励学生直接复制这段，而是理解每个检查为什么存在。'),
        
        ('`TORCH_CHECK` 不是啰嗦，它定义了 op 支持什么、不支持什么。current stream 不是细节，它决定 op 能否正确嵌入 PyTorch 的异步执行和框架调度。',
         '`TORCH_CHECK` 定义了 op 支持什么、不支持什么。current stream 决定 op 能否正确嵌入 PyTorch 的异步执行和框架调度。'),
        
        ('下面代码展示 C++/CUDA extension 的核心结构，支持 `float32` 和 `float16`，并显式处理 current stream 和 launch check。这是从 kernel 到框架算子的完整模板。',
         '下面代码展示 C++/CUDA extension 的核心结构，支持 float32 和 float16，并显式处理 current stream 和 launch check。这是从 kernel 到框架算子的完整模板。'),
        
        ('**关键理解**：Fake kernel 不是"近似计算"，而是"形状推断"。Dynamo 需要知道 `output.shape`、`output.dtype`、`output.device`，才能构建正确的 FX graph。Fake kernel 不分配真实内存，不执行真实计算，但必须在逻辑上完全等价于真实 forward 的 shape 推导。',
         'Fake kernel 不是近似计算，而是形状推断。Dynamo 需要知道 `output.shape`、`output.dtype`、`output.device`，才能构建正确的 FX graph。Fake kernel 不分配真实内存，不执行真实计算，但必须在逻辑上等价于真实 forward 的 shape 推导。'),
        
        ('**为什么 `register_autograd` 是 modern way？**',
         '为什么 `register_autograd` 是现代推荐做法？'),
        
        ('- `torch.compile` 把 custom op 当作 graph 中的一个 opaque 节点，不需要 trace 到 backward 内部。',
         '- `torch.compile` 把 custom op 当作 graph 中的 opaque 节点，不需要 trace 到 backward 内部。'),
        
        ('- 对比 `torch.autograd.Function`：Dynamo 会尝试 trace 进 `forward()` 方法体，如果里面有 opaque kernel 调用，就会 graph break。',
         '- 对比 `torch.autograd.Function`：Dynamo 会尝试 trace 进 `forward()` 方法体，里面有 opaque kernel 调用时就会 graph break。'),
        
        ('`torch.compile`（Dynamo + AOTAutograd + Inductor）在编译时需要遍历整个计算图。如果遇到一个 custom op 且不知道它的输出形状，Dynamo 会在该位置 graph break，导致：',
         '`torch.compile`（Dynamo + AOTAutograd + Inductor）编译时需要遍历整个计算图。遇到 custom op 但不知道输出形状时，Dynamo 会 graph break，导致：'),
        
        ('从源码到可导入的 Python 模块，有多条构建路径。选择哪条取决于你的迭代速度需求和部署环境。',
         '从源码到可导入的 Python 模块，有多条构建路径。选择哪条取决于迭代速度需求和部署环境。'),
        
        ('JIT 编译是最快的原型路径。你把 C++/CUDA 代码以字符串形式传给 Python，PyTorch 在第一次导入时自动编译、缓存 `.so` 文件。',
         'JIT 编译是最快的原型路径。把 C++/CUDA 代码以字符串形式传给 Python，PyTorch 在第一次导入时自动编译、缓存 `.so` 文件。'),
        
        ('AOT 编译是 production 的标准路径。你需要写 `setup.py`（或 `pyproject.toml`），定义 `CUDAExtension` 和 `BuildExtension`。',
         'AOT 编译是 production 的标准路径。需要写 `setup.py`（或 `pyproject.toml`），定义 `CUDAExtension` 和 `BuildExtension`。'),
        
        ('`TORCH_CUDA_ARCH_LIST` 是关键环境变量。如果不设置，PyTorch 会为所有检测到的 GPU 以及 PTX 编译，可能导致编译时间极长。生产环境应显式指定目标架构。',
         '`TORCH_CUDA_ARCH_LIST` 是关键环境变量。不设置时，PyTorch 会为所有检测到的 GPU 以及 PTX 编译，编译时间可能极长。生产环境应显式指定目标架构。'),
        
        ('如果你需要把 PyTorch extension 集成到更大的 C++ 项目（如一个既有 C++ 库又想暴露 PyTorch tensor 接口），CMake 是更灵活的选择。',
         '如果需要把 PyTorch extension 集成到更大的 C++ 项目（如一个既有 C++ 库又想暴露 PyTorch tensor 接口），CMake 是更灵活的选择。'),
        
        ('CMake 路径适合：',
         'CMake 路径适合：'),
        
        ('CUDA extension 的 wheel 是**平台相关**的（Linux x86_64 + CUDA 12.x + torch 2.x）。发布时需要在 CI 中构建多平台 wheel，或使用 `torch.utils.cpp_extension` 的 JIT 路径避免分发二进制。',
         'CUDA extension 的 wheel 是平台相关的（Linux x86_64 + CUDA 12.x + torch 2.x）。发布时需要在 CI 中构建多平台 wheel，或用 JIT 路径避免分发二进制。'),
        
        ('vLLM 的 custom op 不是普通 PyTorch extension 的简单包装。它处在推理系统里，常常需要按 platform、backend、dtype、模型结构选择不同实现。vLLM 提供了 `CustomOp` 基类和 `@CustomOp.register` 装饰器来统一管理这种多态分发。',
         'vLLM 的 custom op 不是普通 PyTorch extension 的简单包装。它处在推理系统里，需要按 platform、backend、dtype、模型结构选择不同实现。vLLM 提供 `CustomOp` 基类和 `@CustomOp.register` 装饰器来统一管理多态分发。'),
        
        ('vLLM 支持 out-of-tree (OOT) 设备插件。假设你是一家加速器厂商，想让 vLLM 跑在你的硬件上：',
         'vLLM 支持 out-of-tree (OOT) 设备插件。假设你是一家加速器厂商，想让 vLLM 跑在你的硬件上：'),
        
        ('`__new__` 方法的魔力：当 vLLM 内部实例化 `FusedRMSNorm` 时，如果 `op_registry_oot` 中有同名的 OOT 类，实际实例化的是 OOT 版本。这实现了**运行时替换**，无需修改 vLLM 源码。',
         '`__new__` 的魔力：vLLM 内部实例化 `FusedRMSNorm` 时，如果 `op_registry_oot` 中有同名 OOT 类，实际实例化的是 OOT 版本。这实现运行时替换，无需修改 vLLM 源码。'),
        
        ('推理框架经常用 CUDA graph 减少 launch overhead。CUDA graph 要求：',
         '推理框架经常用 CUDA graph 减少 launch overhead。CUDA graph 要求：'),
        
        ('vLLM 的 `CustomOp` 如何应对：',
         'vLLM 的 `CustomOp` 如何应对：'),
        
        ('vLLM 通过 `current_platform` 对象封装硬件差异：',
         'vLLM 通过 `current_platform` 对象封装硬件差异：'),
        
        ('vLLM 的 `CompilationConfig` 决定哪些 custom op 启用、哪些禁用：',
         'vLLM 的 `CompilationConfig` 决定哪些 custom op 启用、哪些禁用：'),
        
        ('用户可以通过启动参数控制：',
         '用户可以通过启动参数控制：'),
        
        ('SGLang 和 vLLM 一样面向高性能 LLM serving，但调度、后端、kernel 组织方式有所不同。本节聚焦于 SGLang 的 custom backend 和 attention 机制，帮助理解 custom op 在 SGLang 中的落点。',
         'SGLang 和 vLLM 一样面向高性能 LLM serving，但调度、后端、kernel 组织方式不同。本节聚焦 SGLang 的 custom backend 和 attention 机制，帮助理解 custom op 在 SGLang 中的落点。'),
        
        ('SGLang 的核心创新是 `RadixAttention`：它把所有已计算过的 token 序列组织成一棵 **radix tree**（压缩前缀树），树的每条边对应一段 token 序列，值是对应的 KV cache 张量句柄。',
         'SGLang 的核心创新是 `RadixAttention`：它把所有已计算过的 token 序列组织成一棵 radix tree（压缩前缀树），树的每条边对应一段 token 序列，值是对应的 KV cache 张量句柄。'),
        
        ('当新请求到达时，SGLang 在 radix tree 中做**最长前缀匹配**：',
         '新请求到达时，SGLang 在 radix tree 中做最长前缀匹配：'),
        
        ('这与 vLLM 的 `PagedAttention` 对比：',
         '这与 vLLM 的 `PagedAttention` 对比：'),
        
        ('对 custom op 开发者的启示：',
         '对 custom op 开发者的启示：'),
        
        ('SGLang 支持通过 backend 插件接入自定义 attention / sampling / 量化算子。与 vLLM 的 `CustomOp` 不同，SGLang 的 backend 更偏向**子图替换**：',
         'SGLang 支持通过 backend 插件接入自定义 attention / sampling / 量化算子。与 vLLM 的 `CustomOp` 不同，SGLang 的 backend 更偏向子图替换：'),
        
        ('SGLang 的 backend 抽象允许：',
         'SGLang 的 backend 抽象允许：'),
        
        ('课程里不要求把所有框架内部机制背下来，而是训练一个**迁移能力**：任何推理框架里 custom kernel 最终都要回答四个问题。',
         '课程不要求背框架内部机制，而是训练迁移能力：任何推理框架里 custom kernel 最终都要回答四个问题。'),
        
        ('1. 这个算子被谁调用？\n2. 输入 layout 是谁保证的？\n3. 输出会喂给哪个下游算子？\n4. 当硬件、dtype、shape 不支持时，fallback 是什么？',
         '1. 这个算子被谁调用？\n2. 输入 layout 是谁保证的？\n3. 输出会喂给哪个下游算子？\n4. 硬件、dtype、shape 不支持时，fallback 是什么？'),
        
        ('把这四个问题问清楚，比记住某个版本的文件名更重要。',
         '问清楚这四个问题，比记住某个版本的文件名更重要。'),
        
        ('精度验证不是 `torch.allclose` 一句。不同 kernel 的数值误差来自：',
         '精度验证不是 `torch.allclose` 一句。不同 kernel 的数值误差来自：'),
        
        ('推荐报告格式：',
         '推荐报告格式：'),
        
        ('`torch.compile` 使用 Inductor 作为默认 backend，它会尝试将相邻的 pointwise 操作融合成一个 kernel。如果你的 custom op 被标记为 opaque（即 Inductor 看不到内部），它就像一个"黑盒子"，Inductor 无法跨越它做 fusion。',
         '`torch.compile` 使用 Inductor 作为默认 backend，它会尝试将相邻的 pointwise 操作融合成一个 kernel。如果 custom op 被标记为 opaque（Inductor 看不到内部），它就像一个黑盒子，Inductor 无法跨越它做 fusion。'),
        
        ('优化策略：',
         '优化策略：'),
        
        ('Dynamo 对 custom op 的处理：',
         'Dynamo 对 custom op 的处理：'),
        
        ('LLM 推理有两个阶段：',
         'LLM 推理有两个阶段：'),
        
        ('vLLM 的 `CustomOp` 在 prefill 和 decode 中可能被调用不同的实现：',
         'vLLM 的 `CustomOp` 在 prefill 和 decode 中可能调用不同实现：'),
        
        ('在多 GPU 环境下（Data Parallel / Tensor Parallel / Pipeline Parallel），custom op 必须正确处理：',
         '在多 GPU 环境下（Data Parallel / Tensor Parallel / Pipeline Parallel），custom op 必须正确处理：'),
        
        ('除了 `TORCH_LIBRARY`，你也可以用 pybind11 直接暴露 C++ 函数到 Python。这种方式更灵活，适合需要暴露复杂 C++ 类或非 tensor 接口的场景。',
         '除了 `TORCH_LIBRARY`，也可以用 pybind11 直接暴露 C++ 函数到 Python。这种方式更灵活，适合暴露复杂 C++ 类或非 tensor 接口的场景。'),
        
        ('pybind11 vs TORCH_LIBRARY 的选择：',
         'pybind11 vs TORCH_LIBRARY 的选择：'),
        
        ('**关键点**：',
         '**关键点**'),
        
        ('- `@CustomOp.register("my_fused_mlp")` 把类名注册到全局 `op_registry`。\n- `forward_native` 不是可选的：它是 CUDA graph 不兼容、PyTorch Inductor 启用、或用户显式禁用 custom op 时的 fallback。\n- `forward_cuda` 里调用 `torch.ops.vllm.my_fused_mlp`，这意味着你的 C++ 扩展必须在 `vllm` 命名空间下注册了这个 op。\n- `__new__` 的 OOT 替换机制允许第三方硬件厂商在不修改 vLLM 源码的情况下替换这个 op。',
         '- `@CustomOp.register("my_fused_mlp")` 把类名注册到全局 `op_registry`。\n- `forward_native` 不是可选的：它是 CUDA graph 不兼容、PyTorch Inductor 启用、或用户禁用 custom op 时的 fallback。\n- `forward_cuda` 调用 `torch.ops.vllm.my_fused_mlp`，C++ 扩展必须在 `vllm` 命名空间下注册了这个 op。\n- `__new__` 的 OOT 替换机制允许第三方硬件厂商不修改 vLLM 源码就替换这个 op。'),
        
        ('目标：实现 `scale_add(x, y, alpha)`，输出 `x + alpha * y`。',
         '目标：实现 `scale_add(x, y, alpha)`，输出 `x + alpha * y`。'),
        
        ('要求：',
         '要求：'),
        
        ('这个任务故意简单，因为重点是工程路径，不是 kernel 花活。',
         '这个任务故意简单，重点是工程路径，不是 kernel 花活。'),
        
        ('目标：让 op 能在 fake tensor 或 compile 相关路径下至少完成 shape propagation。',
         '目标：让 op 能在 fake tensor 或 compile 相关路径下完成 shape propagation。'),
        
        ('任务：',
         '任务：'),
        
        ('目标：让 `scale_add` 支持梯度回传。',
         '目标：让 `scale_add` 支持梯度回传。'),
        
        ('任务：',
         '任务：'),
        
        ('为一个假想 `fused_rmsnorm_quant` op 写设计文档：',
         '为假想 `fused_rmsnorm_quant` op 写设计文档：'),
        
        ('提交：',
         '提交：'),
        
        ('- 只写 kernel，不写 shape / dtype / device contract。',
         '- 只写 kernel，不写 shape / dtype / device contract。'),
        
        ('- 认为接入 vLLM / SGLang 只是改一个 Python wrapper。',
         '- 认为接入 vLLM / SGLang 只是改一个 Python wrapper。'),
        
        ('- 没有 fallback，导致不支持的 GPU 上直接崩溃。',
         '- 没有 fallback，导致不支持的 GPU 上直接崩溃。'),
        
        ('- 忽略 `torch.compile` 的 graph break 问题，导致性能不升反降。',
         '- 忽略 `torch.compile` 的 graph break 问题，导致性能不升反降。'),
        
        ('- benchmark 不 warmup，测到的是 JIT 编译时间而非真实性能。',
         '- benchmark 不 warmup，测到的是 JIT 编译时间而非真实性能。'),
        
        ('- 对 FP16 / BF16 使用 FP32 的 tolerance，导致误报或漏报。',
         '- 对 FP16 / BF16 使用 FP32 的 tolerance，导致误报或漏报。'),
        
        ('**一个 CUDA 算子只有完成 framework contract、correctness、benchmark、fallback 和 packaging，才从实验代码变成工程组件。**',
         '一个 CUDA 算子只有完成 framework contract、correctness、benchmark、fallback 和 packaging，才从实验代码变成工程组件。'),
        
        ('本课必须以 PyTorch 官方 custom op 教程和 vLLM CustomOp 文档为主。因为现代算子工程的难点不只是 CUDA kernel，而是 schema、dispatch、fake/meta implementation、current stream、benchmark、fallback，以及推理框架如何调用这个 op。',
         '本课以 PyTorch 官方 custom op 教程和 vLLM CustomOp 文档为主。现代算子工程的难点不只是 CUDA kernel，而是 schema、dispatch、fake/meta implementation、current stream、benchmark、fallback，以及推理框架如何调用这个 op。'),
    ]
    
    for old_str, new_str in replacements:
        if old_str in text:
            text = text.replace(old_str, new_str)
            changes.append(f"替换: {old_str[:50]}...")
    
    return text, changes

def main(ctx):
    with open('/Users/bowenyuchi/Documents/cuda-course/modules/module-17-pytorch-vllm-sglang-custom-op-integration.md', 'r') as f:
        content17 = f.read()

    new_content17, changes17 = humanize_module17(content17)
    
    with open('/Users/bowenyuchi/Documents/cuda-course/modules/module-17-pytorch-vllm-sglang-custom-op-integration.md', 'w') as f:
        f.write(new_content17)
    
    return {"module17_changes": len(changes17), "sample_changes": changes17[:10]}

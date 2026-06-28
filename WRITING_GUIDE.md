# 写作与学习指南

## 课程语气

- 用中文解释概念，保留 CUDA API、工具、kernel、thread、block、warp、stream 等英文术语。
- 先讲清 mental model，再写代码，再测量。
- 避免“这个优化一定更快”这类绝对说法。性能结论必须绑定硬件、输入规模、编译选项和测量工具。
- 每一课都要有故事线。故事线必须解释“为什么现在学这个”，而不是只列“本节内容”。
- 每一课都要有可记忆的类比，但类比之后必须回到硬件或代码事实。
- 每一课都要有一句可背下来的核心句，帮助学习者形成长期记忆。

## 代码风格

- 初学阶段使用朴素 CUDA C++，避免复杂模板和宏技巧。
- host code 必须包含 CUDA 错误检查。
- kernel 必须显式写 bounds check，除非题目正在讨论越界或性能 tradeoff。
- 浮点结果检查必须写 tolerance。
- benchmark 必须包含 warmup，且区分 H2D、kernel、D2H。

## 每节课结构

每个模块都应包含：

- 这一课的故事线
- 一个类比或记忆钩子
- 从硬件视角看问题
- 学习目标
- mental model
- minimal working example 或明确实验目标
- lab 步骤
- 调试或 profiling 任务
- 练习阶梯：回忆、追踪、修改、实现、优化、解释
- checkpoint
- 常见错误
- extension
- 本课要记住的一句话

## 深入度标准

一节 CUDA 课如果只讲 API，不合格。每一课至少要回答：

- 这个 API 或模式解决什么硬件或并行问题？
- 它和 SM、warp、memory hierarchy、copy engine、register、shared memory 或 profiler 中的哪个事实有关？
- 初学者最容易形成什么错误心智模型？
- 如何用一个小实验验证讲义中的说法？
- 这个结论在哪些硬件或输入规模下可能失效？

## 精品课质量门槛

每一课都要按大学讲义或高质量技术博客标准写。具体要求：

- 单课讲义文件总长度必须超过 3000 字符；写作目标应明显高于 3000，避免刚好压线。
- 先查一手资料，再写讲义。优先 NVIDIA 官方文档、PyTorch/vLLM/SGLang/Triton/TileLang 官方文档、DeepSeek 官方仓库、论文或源码。
- 每一课必须在本课内列出资料来源，不能只依赖总资源表。
- 每一课至少包含 5 个层次：问题背景、直觉类比、硬件机制、代码路径、真实系统落点。
- 讲硬件时不能只说“GPU 有很多线程”。要讲到 SM、warp、register、shared memory、L2、HBM、copy engine、Tensor Core、NVLink/RDMA 中和本课相关的部分。
- 讲高级主题时必须说明架构代际：Volta、Turing、Ampere、Hopper、Blackwell 哪些能力变化会影响写法。
- 每一课必须包含至少一段注释充分的精品代码。代码可以是 CUDA C++、CMake、PyTorch extension、Triton、TileLang 或结构化伪代码，但必须服务本课核心概念。
- 每个性能结论必须有 profiler 或 benchmark 设计，不允许只写“更快”。
- 每个现实框架章节必须说明注册、测试、精度对齐、性能对比、fallback 和 CI/packaging。
- 每个模块要有图。优先 Mermaid 直接嵌入 Markdown；复杂系统图可补 Excalidraw。
- 文字宁可详细，不要跳步。每个新术语首次出现时都要解释它解决什么问题。

验收命令建议：

```bash
wc -m modules/*.md
rg -n "Sources|资料来源|精品代码|Checkpoint" modules
```

## Profiling Report 模板

```markdown
Hypothesis:
Hardware:
CUDA Toolkit:
Input sizes:
Build flags:
Measurement tool:
Baseline result:
Bottleneck evidence:
Change:
New result:
Conclusion:
Next experiment:
```

## 图示策略

课程图示保存在 `diagrams/`。`.excalidraw` 是可编辑源文件，`diagrams/exported/*.png` 是嵌入 Markdown 的渲染版。当需要新增图示时，使用 `$excalidraw-diagram-generator` 生成 `.excalidraw` 文件，再导出 PNG 并在讲义中用 Markdown 图片语法嵌入。

字体要求：本课程所有 Excalidraw 图必须使用 Excalifont / 手写风。具体到当前 Excalidraw JSON，所有文本元素统一使用 `fontFamily: 5`。不要把图中文字改成浏览器 sans-serif 风格的 `fontFamily: 1`；如果编辑器或脚本把字体改掉，导出前必须改回 `fontFamily: 5`。

导出时不要直接用 `@moona3k/excalidraw-export` 生成最终 PNG，因为它不会正确处理 `fontFamily: 5`。应使用课程内的 `scripts/export_excalidraw_with_excalifont.mjs`，该脚本会加载 `assets/fonts/Excalifont-Regular.ttf`，并用 `assets/fonts/LXGWWenKai-Regular.ttf` 作为中文手写/文楷 fallback。

每个模块标题下应放一张主图，并在图片下方写一句说明和源图链接。图片说明要解释这张图帮助读者建立哪种心智模型，而不是只重复文件名。

字体检查命令：

```bash
python3 scripts/check_diagram_fonts.py
```

当前导出命令示例：

```bash
node scripts/export_excalidraw_with_excalifont.mjs
```

优先画这些图：

- CUDA 执行模型
- CUDA 内存层级
- profiling 优化闭环
- stream/event 并发时间线
- 课程知识图谱
- capstone 系统数据流

图示必须解释学习难点，不能只是装饰。

## 答案与提示

- 诊断题可以附答案。
- 每周练习默认只给提示，不直接给完整代码。
- 项目和 capstone 默认不给完整解法，除非学习者明确要求。

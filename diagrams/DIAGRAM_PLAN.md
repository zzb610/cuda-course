# Diagram Plan

这些图示可用 Excalidraw 打开，也可继续用 `$excalidraw-diagram-generator` 修改或重画。`.excalidraw` 文件是源图，`diagrams/exported/*.png` 是已经嵌入 Markdown 讲义的渲染版。

字体策略：课程图统一使用 Excalifont / 手写风。当前所有源图文本应使用 `fontFamily: 5`，导出后呈现 Excalidraw 的手写字体风格。不要切换到浏览器 sans-serif 风格的 `fontFamily: 1`。最终 PNG 必须用 `scripts/export_excalidraw_with_excalifont.mjs` 导出；不要直接用 `@moona3k/excalidraw-export` 作为最终导出命令，因为它不会正确处理 `fontFamily: 5`。

修改或新增图后先运行：

```bash
python3 scripts/check_diagram_fonts.py
```

重新导出所有图片：

```bash
node scripts/export_excalidraw_with_excalifont.mjs
```

## Generated Diagrams

- `course-knowledge-map.excalidraw`: 课程知识图谱，用于 Module 00、06、07。
- `cuda-execution-model.excalidraw`: host/device、grid/block/thread/warp 关系，用于 Module 01、04。
- `cuda-memory-hierarchy.excalidraw`: CUDA 内存层级，用于 Module 02、03、09。
- `profiling-optimization-loop.excalidraw`: profiling 优化闭环，用于 Module 05。
- `stream-concurrency-timeline.excalidraw`: streams/events/overlap 时间线，用于 Module 08。
- `capstone-system-pipeline.excalidraw`: capstone 工程数据流，用于 Module 10、11。

## Regeneration Prompts

### CUDA Execution Model

Use `$excalidraw-diagram-generator` to create a relationship diagram showing CPU host, CUDA runtime, kernel launch, GPU device, grid, blocks, threads, warps, and host/device memory copies. Keep it under 15 elements.

### CUDA Memory Hierarchy

Use `$excalidraw-diagram-generator` to create a relationship diagram showing registers, local memory, shared memory, global memory, constant memory, and host memory, with notes about scope and latency intuition.

### Profiling Optimization Loop

Use `$excalidraw-diagram-generator` to create a flowchart: hypothesis -> baseline -> measure -> identify bottleneck -> change one thing -> re-measure -> document result -> next hypothesis.

### Stream Concurrency Timeline

Use `$excalidraw-diagram-generator` to create a sequence/timeline diagram with host enqueue, stream 0 H2D/kernel/D2H, stream 1 H2D/kernel/D2H, events, and synchronization.

### Course Knowledge Map

Use `$excalidraw-diagram-generator` to create a mind map centered on CUDA mastery with branches: correctness, memory, profiling, algorithms, libraries, architecture-aware optimization, production engineering.

### Capstone System Pipeline

Use `$excalidraw-diagram-generator` to create a data-flow diagram: input data -> CPU orchestration -> H2D copy -> custom kernels/library calls -> D2H results -> tests/benchmark/profiling report.

### GPU Architecture Evolution

Use `$excalidraw-diagram-generator` to create a timeline diagram from Volta to Blackwell. Include Tensor Core evolution, Ampere cp.async, Hopper TMA/WGMMA/thread block clusters, Blackwell FP4/FP8 evolution, and SM100-specific tcgen05/NVLink 5 notes. Keep it source-grounded and avoid exact numeric specs unless checked.

### Tensor Core GEMM Stack

Use `$excalidraw-diagram-generator` to create a layered diagram: PyTorch matmul -> cuBLAS/CUTLASS/DeepGEMM -> CTA tile -> warp/warpgroup tile -> MMA/WGMMA instruction -> Tensor Core. Include dtype/layout/epilogue as side annotations.

### TMA WGMMA Pipeline

Use `$excalidraw-diagram-generator` to create a sequence diagram showing global memory -> TMA async copy -> shared memory tile -> mbarrier wait -> WGMMA compute -> epilogue -> global memory. Show producer/consumer stages.

### MoE DeepEP Communication

Use `$excalidraw-diagram-generator` to create a system diagram for MoE: tokens -> router -> dispatch -> NVLink/RDMA -> expert GEMM -> combine -> restore order. Include normal throughput mode and low-latency decode mode as two lanes.

### PyTorch To vLLM Custom Op

Use `$excalidraw-diagram-generator` to create a flowchart: CUDA kernel -> C++ launcher -> TORCH_LIBRARY binding -> torch.ops Python wrapper -> PyTorch tests/benchmark -> vLLM CustomOp/SGLang integration -> serving scheduler.

### PagedAttention KV Cache

Use `$excalidraw-diagram-generator` to create a relationship diagram showing logical sequence tokens, block table, physical KV cache pages, decode query, attention kernel, and output. Make page indirection visually obvious.

### CUDA Triton TileLang Comparison

Use `$excalidraw-diagram-generator` to create a comparison diagram for the same operator implemented in CUDA C++, CUTLASS/CuTe, Triton, and TileLang. Show tradeoff axes: control, development speed, portability, debug cost, framework integration.

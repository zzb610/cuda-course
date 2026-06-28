# Diagnostic Assessment

Use before Module 00. 目标是调整学习路径，不是筛人。

## Part A: C/C++ 基础

1. `sizeof(int*)` 和数组元素个数有什么关系？
2. 为什么函数收到 `float* p` 后通常不知道数组长度？
3. row-major 的 `A[row][col]` 在线性内存里如何计算 offset？
4. `const float*` 和 `float* const` 有什么区别？
5. CMake 的 configure 和 build 大致有什么区别？

## Part B: 并行思维

1. 什么样的 loop 容易并行化？
2. race condition 是什么？
3. 如果 1024 个线程同时写同一个地址，会发生什么？
4. 为什么“更多线程”不一定更快？

## Part C: CUDA 起点

1. GPU kernel 是在 CPU 还是 GPU 上执行？
2. host memory 和 device memory 能否直接混用？
3. 为什么 kernel launch 后要检查错误？
4. 你能否运行 `nvidia-smi` 和 `nvcc --version`？

## Suggested Routing

- A 部分错 3 个以上：先补 C/C++ 指针、数组、编译。
- B 部分错 2 个以上：Module 01 和 04 放慢速度。
- C 部分无法运行工具：先完成环境修复，不进入 CUDA labs。

## Short Answer Key

### Part A

1. `sizeof(int*)` 是指针变量本身的大小，不是数组元素个数。
2. `float* p` 只保存起始地址，不携带长度；长度通常要单独传入。
3. row-major 下 offset 通常是 `row * width + col`。
4. `const float*` 表示指向只读 float 的指针；`float* const` 表示指针变量本身不能改指向。
5. configure 生成构建系统和缓存配置，build 按生成结果编译目标。

### Part B

1. 各次迭代之间没有写后读、写后写等跨迭代依赖的 loop 更容易并行化。
2. race condition 是多个执行单元并发读写共享状态，最终结果取决于不可控的执行时序。
3. 1024 个线程同时写同一地址通常是数据竞争；最终值不确定，除非使用 atomic 或有明确定义的规约/同步方案。
4. 更多线程可能增加调度、同步、内存带宽、寄存器和 shared memory 压力；性能取决于瓶颈，不只取决于线程数量。

### Part C

1. GPU kernel 在 GPU device 上执行；host 端代码负责 launch、数据准备和同步。
2. 普通 host pointer 和 device pointer 不能随意混用；需要明确拷贝、mapped/pinned memory、managed memory 或框架 tensor 约定。
3. kernel launch 是异步的，启动参数错误和设备端执行错误可能延后暴露；必须检查 launch error，并在需要时同步检查运行错误。
4. `nvidia-smi` 说明系统能看到 NVIDIA driver/GPU；`nvcc --version` 说明 CUDA Toolkit 编译器可用。二者缺一不可直接进入需要本地编译和运行的 CUDA labs。

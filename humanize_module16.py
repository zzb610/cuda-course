import re

def humanize_module16(text):
    changes = []
    
    # 删除记忆钩子
    old = '记忆钩子：**单 GPU 优化看"算得快不快"，多 GPU 优化还要看"喂得上喂不上"。**'
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
        ('前面你已经能写单 GPU kernel，也知道现代 GEMM 如何吃 Tensor Core。现在把场景放到 LLM 推理：模型太大，expert 太多，batch 来得太快，一张 GPU 不够了。你很快会遇到一个残酷事实：多 GPU 系统里，算力经常不是唯一瓶颈，甚至不是第一瓶颈。token 从哪里来，经过哪张网卡，走 PCIe 还是 NVLink，什么时候跨节点，什么时候同步，都会改变最终吞吐和延迟。',
         '前面已经能写单 GPU kernel，也知道现代 GEMM 如何用 Tensor Core。现在看 LLM 推理场景：模型太大，expert 太多，batch 来得太快，一张 GPU 不够。多 GPU 系统里，算力经常不是唯一瓶颈，甚至不是第一瓶颈。token 从哪里来，经过哪张网卡，走 PCIe 还是 NVLink，什么时候跨节点，什么时候同步，都会改变最终吞吐和延迟。'),
        
        ('这节课要把通信当成 GPU 编程的一部分，而不是分布式训练课的附属内容。NCCL 解决的是 GPU 之间高性能 collective 和 point-to-point 通信；NVLink/NVSwitch 解决的是节点内 GPU 间高速互联；GPUDirect RDMA 解决的是网卡和 GPU memory 之间尽量少绕 CPU 的跨节点路径；DeepEP 则把这些能力落到 MoE expert parallel 的 dispatch/combine 这类真实推理工作负载上。',
         '这节课把通信当成 GPU 编程的一部分。NCCL 解决 GPU 之间高性能 collective 和 point-to-point 通信；NVLink/NVSwitch 解决节点内 GPU 间高速互联；GPUDirect RDMA 解决网卡和 GPU memory 之间尽量少绕 CPU 的跨节点路径；DeepEP 把这些能力落到 MoE expert parallel 的 dispatch/combine 上。'),
        
        ('如果只学 CUDA kernel，你会问："我的 kernel occupancy 够不够？"学到这一课，你要同时问："我的 token 搬运路径是不是把 GPU 算子饿住了？"',
         '只学 CUDA kernel 时，问题可能是"kernel occupancy 够不够"。学到这一课，要问"token 搬运路径是不是把 GPU 算子饿住了？"'),
        
        ('把每张 GPU 想成一个大型工厂，把 HBM 想成工厂仓库，把 SM 想成生产线。单 GPU 优化是在工厂内部优化传送带和工人排班；多 GPU 推理是在多个工厂之间调货。',
         '每张 GPU 是工厂，HBM 是仓库，SM 是生产线。单 GPU 优化是工厂内部优化；多 GPU 推理是多个工厂之间调货。'),
        
        ('- **PCIe** 像城市普通道路，通用但不一定最快。所有车辆（GPU、网卡、NVMe）都走这条路，会遇到红绿灯和拥堵。\n- **NVLink** 像工厂园区里的专用高速路，点对点直达，限速高、没有红绿灯。\n- **NVSwitch** 像高速路立交，让多座工厂互相直连得更均衡——从任何工厂到任何工厂都是同样的高速距离。\n- **RDMA** 像给跨城货运开了专线，货车可以把货直接送到目标工厂仓库附近，不需要先卸到城市物流中心（CPU memory）。\n- **NCCL** 像调度系统，知道不同货运模式该怎么排路线、怎么编车队、怎么在多个工厂间同步卸货。\n- **DeepEP** 像专门给 MoE 餐厅设计的送餐系统：先把 token 送到对应 expert，算完再组合回来。它知道哪些订单急（decode）、哪些订单量大（prefill），用不同的配送策略。',
         '- **PCIe** 像城市普通道路，所有车辆都走这条路，会拥堵。\n- **NVLink** 像工厂园区里的专用高速路，点对点直达，限速高。\n- **NVSwitch** 像高速路立交，让多座工厂互相直连。从任何工厂到任何工厂都是同样高速距离。\n- **RDMA** 像给跨城货运开专线，货车直接把货送到目标仓库附近，不需要先卸到城市物流中心（CPU memory）。\n- **NCCL** 像调度系统，知道不同货运模式怎么排路线、编车队、在多个工厂间同步卸货。\n- **DeepEP** 像给 MoE 餐厅设计的送餐系统：先把 token 送到对应 expert，算完再组合回来。它知道 decode 订单急、prefill 订单量大，用不同配送策略。'),
        
        ('注意：即使是最快的 NVLink 5.0，带宽/算力比仍在下降。这意味着**通信越来越容易成为瓶颈**。',
         '即使是最快的 NVLink 5.0，带宽/算力比仍在下降。通信越来越容易成为瓶颈。'),
        
        ('MoE 的 dispatch/combine 是最棘手的：消息大小随 batch 和 top-k 动态变化，不同 expert 的负载极不均衡，而且每个 MoE 层都要执行两次。',
         'MoE 的 dispatch/combine 最棘手：消息大小随 batch 和 top-k 动态变化，不同 expert 负载极不均衡，而且每个 MoE 层都要执行两次。'),
        
        ('这张图很重要。很多初学者把 communication library 当成 CPU 侧 API，但真实路径会穿过 CUDA stream、GPU memory、PCIe/NVLink、网卡、拓扑发现、buffer registration、kernel launch 甚至 CUDA graph。调性能时你必须知道自己卡在哪一层。',
         '很多初学者把 communication library 当成 CPU 侧 API，但真实路径会穿过 CUDA stream、GPU memory、PCIe/NVLink、网卡、拓扑发现、buffer registration、kernel launch 甚至 CUDA graph。调性能时要知道自己卡在哪一层。'),
        
        ('对比 PCIe 5.0 x16 的 ~128 GB/s，NVLink 4.0 的 900 GB/s 是 **7 倍**。更重要的是，NVLink 是点对点直连，避免了 PCIe 的共享总线仲裁和 CPU 中心式瓶颈。',
         'PCIe 5.0 x16 约 128 GB/s，NVLink 4.0 是 900 GB/s，约 7 倍。NVLink 是点对点直连，避免了 PCIe 的共享总线仲裁和 CPU 中心式瓶颈。'),
        
        ('NVSwitch 是一个专用交换 ASIC，功能类似于网络交换机，但专门为 NVLink 设计。它让多 GPU 之间形成**全互联（full mesh / non-blocking fabric）**。',
         'NVSwitch 是专用交换 ASIC，类似网络交换机，但专门为 NVLink 设计。它让多 GPU 之间形成全互联。'),
        
        ('**关键洞察**：在 NVSwitch 系统里，任意两个 GPU 之间的带宽和延迟是均匀的。这意味着 tensor parallelism 的 all-reduce 性能不随 GPU 对的选择而变化。但在没有 NVSwitch 的系统里（比如 PCIe 直连 GPU），GPU 0 到 GPU 1 可能走 PCIe switch，GPU 0 到 GPU 4 可能跨 CPU socket，带宽和延迟完全不同。',
         '在 NVSwitch 系统里，任意两个 GPU 之间的带宽和延迟是均匀的。tensor parallelism 的 all-reduce 性能不随 GPU 对选择变化。没有 NVSwitch 的系统（比如 PCIe 直连 GPU），GPU 0 到 GPU 1 可能走 PCIe switch，GPU 0 到 GPU 4 可能跨 CPU socket，带宽和延迟完全不同。'),
        
        ('CUDA 提供了 `cudaDeviceCanAccessPeer` 来检查两个 GPU 是否可以直接访问对方的内存。这是硬件拓扑决定的：如果两个 GPU 通过同一个 PCIe switch 或 NVLink 连接，通常返回 true；如果跨 CPU socket 或跨不同 PCIe root complex，可能返回 false。',
         'CUDA 提供 `cudaDeviceCanAccessPeer` 检查两个 GPU 是否可以直接访问对方内存。这是硬件拓扑决定：如果两个 GPU 通过同一个 PCIe switch 或 NVLink 连接，通常返回 true；跨 CPU socket 或不同 PCIe root complex，可能返回 false。'),
        
        ('UVA 是 CUDA 4.0 引入的特性，要求 64-bit 操作系统、Fermi 及以上架构。在 UVA 下，CPU 和所有 GPU 的内存共享同一个虚拟地址空间。这意味着：',
         'UVA 是 CUDA 4.0 引入的特性，要求 64-bit 操作系统、Fermi 及以上架构。在 UVA 下，CPU 和所有 GPU 内存共享同一个虚拟地址空间。这意味着：'),
        
        ('但注意：P2P 访问的实际性能取决于物理路径。如果两个 GPU 只通过 PCIe 连接，kernel 直接访问 remote GPU 的内存可能比显式 `cudaMemcpyPeer` 更慢，因为 PCIe 的延迟对随机访问不友好。NVLink 由于带宽高、延迟低，更适合 P2P kernel 访问。',
         'P2P 访问的实际性能取决于物理路径。两个 GPU 只通过 PCIe 连接时，kernel 直接访问 remote GPU 内存可能比显式 `cudaMemcpyPeer` 更慢，因为 PCIe 延迟对随机访问不友好。NVLink 带宽高、延迟低，更适合 P2P kernel 访问。'),
        
        ('RDMA（Remote Direct Memory Access）允许一台机器的网卡直接读写另一台机器的内存，无需 CPU 参与。三种主要实现：',
         'RDMA（Remote Direct Memory Access）允许一台机器的网卡直接读写另一台机器内存，无需 CPU 参与。三种实现：'),
        
        ('RDMA 需要知道内存的物理地址才能编程 DMA。但 GPU 内存由 GPU driver 的虚拟内存管理器（VMM）管理，物理页可能不连续，甚至可能被换出。因此需要**注册（registration）**或**锁定（pinning）**：',
         'RDMA 需要知道内存物理地址才能编程 DMA。但 GPU 内存由 VMM 管理，物理页可能不连续，甚至被换出。所以需要注册或锁定：'),
        
        ('问题：这个模块需要针对每个内核版本和 MOFED 版本重新编译，版本不匹配会产生 `Invalid argument` 错误。在 Ubuntu 的 inbox `rdma-core` 栈上，这个 API 不存在，因此模块无法加载。',
         '这个模块需要针对每个内核版本和 MOFED 版本重新编译，版本不匹配会产生 `Invalid argument` 错误。Ubuntu 的 inbox `rdma-core` 没有这个 API，模块无法加载。'),
        
        ('GPUDirect P2P 允许同一节点内的 GPU 直接访问彼此的内存，不经过 CPU 内存。它依赖 PCIe 或 NVLink 的 P2P 能力。注意：',
         'GPUDirect P2P 允许同一节点内 GPU 直接访问彼此内存，不经过 CPU 内存。依赖 PCIe 或 NVLink 的 P2P 能力。注意：'),
        
        ('GPUDirect Storage 允许 GPU 直接从 NVMe SSD 读取数据到 GPU memory，不经过 CPU memory。这对数据加载繁重的训练场景（如视频、大规模图像数据集）很有价值。但它属于另一个话题，本模块不深入。',
         'GPUDirect Storage 允许 GPU 直接从 NVMe SSD 读取数据到 GPU memory，不经过 CPU memory。对数据加载繁重的训练场景（如视频、大规模图像数据集）有价值。但本模块不深入。'),
        
        ('Memory registration 是有开销的：',
         'Memory registration 有开销：'),
        
        ('NCCL 内部就有 registration cache。DeepEP 的 buffer management 也需要考虑注册开销，尤其在使用 RDMA 时。',
         'NCCL 内部有 registration cache。DeepEP 的 buffer management 也要考虑注册开销，尤其使用 RDMA 时。'),
        
        ('NCCL（NVIDIA Collective Communications Library）是面向 NVIDIA GPU 的高性能通信库。核心概念：',
         'NCCL（NVIDIA Collective Communications Library）是面向 NVIDIA GPU 的高性能通信库。核心概念：'),
        
        ('NCCL 的 in-place 语义：`sendbuf` 可以是 `recvbuf` 的地址，但要求数据类型和对齐满足条件。对于 all-reduce，in-place 是最常见用法：',
         'NCCL 的 in-place 语义：`sendbuf` 可以是 `recvbuf` 地址，但要求数据类型和对齐满足条件。all-reduce 的 in-place 最常见：'),
        
        ('NCCL 操作是**异步**的，它们被 enqueue 到 CUDA stream 上，和 kernel 一样遵守 stream ordering。这意味着：',
         'NCCL 操作是异步的，它们被 enqueue 到 CUDA stream，和 kernel 一样遵守 stream ordering。这意味着：'),
        
        ('NCCL 的 internal 实现也使用 CUDA stream。你可以在 NCCL 调用中传入一个 stream，让通信和计算在同一个 stream 上串行，或在不同 stream 上并行。',
         'NCCL 内部也使用 CUDA stream。可以在 NCCL 调用中传入 stream，让通信和计算在同一个 stream 上串行，或在不同 stream 上并行。'),
        
        ('**通信量**：每个 rank 发送和接收 `2*(n-1)*S/n`。对大数据量，这接近带宽最优下界。',
         '通信量：每个 rank 发送和接收 `2*(n-1)*S/n`。大数据量下接近带宽最优下界。'),
        
        ('Dense Transformer 每层通常是规整矩阵乘法和 attention。MoE 层不同：每个 token 会被 router 分到一个或多个 expert。假设 expert 分布在多张 GPU 上，那么 token 必须先被 dispatch 到对应 GPU，expert 算完后再 combine 回原来的 token 顺序。',
         'Dense Transformer 每层是规整矩阵乘法和 attention。MoE 层不同：每个 token 会被 router 分到一个或多个 expert。expert 分布在多张 GPU 上时，token 要先 dispatch 到对应 GPU，expert 算完后再 combine 回原来的 token 顺序。'),
        
        ('DeepEP 是 DeepSeek 开源的专门面向 MoE expert parallelism 的通信库。它解决的核心问题是：通用 NCCL all-to-all 对 MoE 的稀疏、动态、不均衡通信模式不够优化。',
         'DeepEP 是 DeepSeek 开源的面向 MoE expert parallelism 的通信库。它解决的核心问题：通用 NCCL all-to-all 对 MoE 的稀疏、动态、不均衡通信模式不够优化。'),
        
        ('适合 prefill、大 batch、token 多、通信量大。目标是把链路吃满，尽量让每次通信有足够大的 payload。它像高速货运，装满一车再发，单位成本低。',
         '适合 prefill、大 batch、token 多、通信量大。目标是把链路吃满，让每次通信有足够大的 payload。像高速货运，装满一车再发，单位成本低。'),
        
        ('你要观察：',
         '观察点：'),
        
        ('适合 decode、小 batch、每步生成一个或少量 token。目标是减少每步等待时间，不一定追求链路峰值带宽。它像急件配送，车不一定装满，但必须快。',
         '适合 decode、小 batch、每步生成少量 token。目标是减少每步等待时间，不一定追求链路峰值带宽。像急件配送，车不一定装满，但必须快。'),
        
        ('你要观察：',
         '观察点：'),
        
        ('这体现了系统级优化的核心思想：**通信的 buffer layout 是为后续计算服务的**。',
         '系统级优化的核心思想：通信的 buffer layout 是为后续计算服务的。'),
        
        ('MPI 是跨节点分布式的标准。MPI + CUDA 的常见模式：',
         'MPI 是跨节点分布式的标准。MPI + CUDA 的常见模式：'),
        
        ('关键注意事项：',
         '关键注意事项：'),
        
        ('PyTorch 的 `torch.distributed` 默认使用 NCCL backend 进行 GPU 通信。关键概念：',
         'PyTorch 的 `torch.distributed` 默认用 NCCL backend 进行 GPU 通信。关键概念：'),
        
        ('PyTorch 的 NCCL backend 会自动处理：',
         'PyTorch 的 NCCL backend 自动处理：'),
        
        ('DDP 的核心是：每个 rank 独立计算梯度，然后使用 `all_reduce` 同步并平均梯度。',
         'DDP 的核心：每个 rank 独立计算梯度，然后用 `all_reduce` 同步并平均梯度。'),
        
        ('DDP 的优化：',
         'DDP 的优化：'),
        
        ('**组合策略（最常见）**：',
         '**组合策略**'),
        
        ('**逐行注释说明**：',
         '**说明**：'),
        
        ('vLLM 支持三种并行：',
         'vLLM 支持三种并行：'),
        
        ('vLLM 的分布式关键设计：',
         'vLLM 的分布式关键设计：'),
        
        ('Megatron-LM 是 NVIDIA 开源的大规模 Transformer 训练框架，定义了业界标准的 tensor parallel 和 pipeline parallel 算法。',
         'Megatron-LM 是 NVIDIA 开源的大规模 Transformer 训练框架，定义了标准的 tensor parallel 和 pipeline parallel 算法。'),
        
        ('DeepSeek-V3 的 671B 参数模型使用了 256 个 experts，top-8 路由。其 EP 设计特点：',
         'DeepSeek-V3 的 671B 参数模型用 256 个 experts，top-8 路由。EP 设计特点：'),
        
        ('**NCCL 调试关键环境变量**：',
         '**NCCL 调试关键环境变量**'),
        
        ('**把 NCCL 当成和 CUDA stream 无关的后台魔法**。NCCL 操作是异步的，但它们 enqueue 到 CUDA stream，遵守 stream ordering。错误地假设它们"立即完成"会导致数据竞争。',
         '**把 NCCL 当成和 CUDA stream 无关的后台魔法**。NCCL 操作是异步的，enqueue 到 CUDA stream，遵守 stream ordering。假设它们"立即完成"会导致数据竞争。'),
        
        ('**只看 GPU 数量，不看拓扑**。8 张 GPU 通过 NVSwitch 全互联和 8 张 GPU 通过 PCIe 树形连接，all-reduce 性能可能差 5-10 倍。',
         '**只看 GPU 数量，不看拓扑**。8 张 GPU 通过 NVSwitch 全互联和 8 张 GPU 通过 PCIe 树形连接，all-reduce 性能可能差 5-10 倍。'),
        
        ('**以为 RDMA 开启后性能自然变好，忽略 buffer registration 和 NUMA**。RDMA 需要 memory registration，频繁分配释放会导致注册开销。NUMA 不对齐会导致跨 socket 内存访问。',
         '**以为 RDMA 开启后性能自然变好，忽略 buffer registration 和 NUMA**。RDMA 需要 memory registration，频繁分配释放导致注册开销。NUMA 不对齐导致跨 socket 内存访问。'),
        
        ('**用训练 all-reduce 的经验直接套 MoE all-to-all**。All-reduce 的消息大小固定、负载均匀；all-to-all 的消息大小动态、可能极度不均匀。需要完全不同的优化策略。',
         '**用训练 all-reduce 的经验直接套 MoE all-to-all**。All-reduce 消息大小固定、负载均匀；all-to-all 消息大小动态、可能极不均匀。需要不同的优化策略。'),
        
        ('**只优化 expert GEMM，不测 dispatch/combine**。在 MoE 中，通信可能占 50-80% 的 layer 时间。不优化通信，再快的 GEMM 也没用。',
         '**只优化 expert GEMM，不测 dispatch/combine**。MoE 中通信可能占 50-80% 的 layer 时间。不优化通信，再快的 GEMM 也没用。'),
        
        ('**忽略 decode 与 prefill 的延迟/吞吐目标差异**。用 prefill 的高吞吐路径做 decode，会导致小消息延迟过高；用 decode 的低延迟路径做 prefill，会导致带宽利用率不足。',
         '**忽略 decode 与 prefill 的延迟/吞吐差异**。用 prefill 高吞吐路径做 decode，小消息延迟过高；用 decode 低延迟路径做 prefill，带宽利用率不足。'),
        
        ('**在 CUDA graph 里使用动态 shape 的 collective**。CUDA graph 要求固定 launch 参数。如果 MoE 的 all-to-all 消息大小随 batch 变化，graph capture 会失败或需要 padding 到最大 size。',
         '**在 CUDA graph 里使用动态 shape 的 collective**。CUDA graph 要求固定 launch 参数。MoE 的 all-to-all 消息大小随 batch 变化时，graph capture 会失败或需要 padding 到最大 size。'),
        
        ('**混淆 P2P 和 RDMA**。P2P 是节点内 GPU-GPU 直接访问；RDMA 是跨节点网卡直接访问。两者都"绕过 CPU"，但硬件路径和编程接口完全不同。',
         '**混淆 P2P 和 RDMA**。P2P 是节点内 GPU-GPU 直接访问；RDMA 是跨节点网卡直接访问。两者都绕过 CPU，但硬件路径和编程接口不同。'),
        
        ('**忽视 IOMMU 对 GPUDirect 的影响**。在虚拟化环境（KVM、VMware）中，IOMMU 可能阻止设备直接访问 GPU 内存。需要配置 VT-d 或禁用 IOMMU。',
         '**忽视 IOMMU 对 GPUDirect 的影响**。虚拟化环境（KVM、VMware）中，IOMMU 可能阻止设备直接访问 GPU 内存。需要配置 VT-d 或禁用 IOMMU。'),
        
        ('**认为 DMA-BUF 和 nvidia-peermem 一样**。DMA-BUF 是 modern Linux 内核的标准机制，不需要额外模块；`nvidia-peermem` 是 legacy 路径，在新内核上容易失败。',
         '**认为 DMA-BUF 和 nvidia-peermem 一样**。DMA-BUF 是 Linux 内核标准机制，不需要额外模块；`nvidia-peermem` 是 legacy 路径，在新内核上容易失败。'),
        
        ('**多 GPU CUDA 工程的关键不是"把数据发出去"，而是让正确的数据在正确时间沿正确硬件路径到达正确 kernel。**',
         '多 GPU CUDA 工程的关键不是"把数据发出去"，而是让正确的数据在正确时间沿正确硬件路径到达正确 kernel。'),
        
        ('*本模块由 CUDA 编程课程团队编写，基于 NVIDIA 官方文档、DeepEP 开源代码、vLLM/Megatron 源码及多篇 arXiv 论文综合整理。所有代码示例假设标准 CUDA + NCCL + MPI 环境。*',
         '*本模块基于 NVIDIA 官方文档、DeepEP 开源代码、vLLM/Megatron 源码及 arXiv 论文整理。代码示例假设标准 CUDA + NCCL + MPI 环境。*'),
    ]
    
    for old_str, new_str in replacements:
        if old_str in text:
            text = text.replace(old_str, new_str)
            changes.append(f"替换: {old_str[:50]}...")
    
    return text, changes

def main(ctx):
    with open('/Users/bowenyuchi/Documents/cuda-course/modules/module-16-nccl-nvlink-gpudirect-rdma-deepep.md', 'r') as f:
        content16 = f.read()

    new_content16, changes16 = humanize_module16(content16)
    
    with open('/Users/bowenyuchi/Documents/cuda-course/modules/module-16-nccl-nvlink-gpudirect-rdma-deepep.md', 'w') as f:
        f.write(new_content16)
    
    return {"module16_changes": len(changes16), "sample_changes": changes16[:10]}

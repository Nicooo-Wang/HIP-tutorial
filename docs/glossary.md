# 术语表

全课程统一用词。HIP/AMD 文档和代码里通用的英文术语保留英文；有通用中文说法的概念用中文。讲义里术语第一次出现时，按“解释”一栏的意思用一句话说明。

“首次讲解位置”指正式解释这个术语的地方；“课程首页”指 `site/index.html`。

## 用词约定

- **wavefront**：正文写全称 wavefront；图内文字、代码注释和口语化的小标签可以简写为 wave。CK-Tile 源码里的 warp 指的就是 wavefront。
- **block**：始终用 block。AMD 文档里的 workgroup、CUDA 的 thread block 只在 Lab 0 §2.2 对照一次。
- **LDS**：始终用 LDS。CUDA 资料里叫 shared memory，只在 Lab 0 §2.2 和 Lab 1 §1.2 对照。
- **显存 / HBM**：讲硬件时用 HBM，讲编程时用显存；global memory 只在 Lab 0 §2.1 对照一次。
- **归约**：用中文“归约”，第一次出现时注英文 reduction。
- **合并访存**：作术语时用“合并访存”，行文中可以说“合并地读 / 写”。
- **shuffle**：概念叫 wavefront shuffle，指具体函数时写 `__shfl_down`。
- **occupancy**：写作 occupancy（占用率）。
- **分块 / tile**：手写 kernel 时叫“分块”；CK-Tile 语境里保留 tile（tile distribution、tile window）。
- **padding**：LDS 每行多开的列叫 LDS padding；`pad_tensor_view` 做的事叫“pad 视图”。
- **distributed tensor**：概念叫 distributed tensor，类型写 `static_distributed_tensor`，口语解释可以说“每个线程的寄存器数组”。

## 硬件与架构

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| MI308X | 本课使用的 AMD Instinct GPU：gfx942 架构，80 个 CU，wavefront 宽 64，每个 CU 64 KiB LDS。注意与 MI300X（304 个 CU）区分。 | Lab 0 §2.1 |
| gfx942 | MI308X 所属的 GPU 架构代号，属于 CDNA3。编译时用 `--offload-arch=gfx942` 指定。 | Lab 0 §5.3 |
| CDNA3 | AMD 数据中心 GPU 的一代架构，gfx942 属于这一代。 | Lab 0 §2.1 |
| XCD | MI308X 上的计算芯粒（Accelerator Complex Die）。每张卡 4 个 XCD，每个 XCD 有 20 个 CU 和 4 MiB L2。 | Lab 0 §2.1 |
| CU | Compute Unit，计算单元。一个 block 整个放在一个 CU 上执行；每个 CU 有 4 个 SIMD 和 64 KiB LDS。 | Lab 0 §2.1 |
| SIMD | CU 里真正执行 wavefront 指令的单元，每个 CU 有 4 个。每个 SIMD 上可以同时驻留多个 wavefront。 | Lab 0 §2.1 |
| wavefront | 硬件一起执行的一组线程，本机 64 个。同一个 wavefront 的线程每次执行同一条指令。 | Lab 0 §2.2 |
| lane | wavefront 里的一个线程，编号 0–63。 | Lab 0 §2.2（图示），Lab 1 §2.1 |
| LDS | Local Data Share，每个 CU 上 64 KiB 的片上存储。同一个 block 的线程共享，在 kernel 里用 `__shared__` 声明。 | Lab 0 §2.1 |
| HBM | GPU 的主存，也就是显存。本机约 192 GiB，所有 CU 都能访问；容量最大，访问延迟也最长。 | Lab 0 §2.1 |
| L1 | 每个 CU 里 32 KiB 的向量数据缓存，由硬件自动管理。 | Lab 0 §2.1 |
| L2 | 每个 XCD 里 4 MiB 的缓存，由该 XCD 上的 CU 共享，硬件自动管理。 | Lab 0 §2.1 |
| Infinity Cache | 整张卡共享的 256 MiB 末级缓存，位于 L2 和 HBM 之间。 | Lab 0 §2.1 |
| VGPR | 向量通用寄存器，每个 lane 私有，是最快的存储。局部标量通常放在 VGPR 里，但实际位置由编译器决定。 | Lab 1 §1.2 |
| SGPR | 标量通用寄存器，一个 wavefront 的所有 lane 共用一份，常用来存 kernel 参数这类对所有 lane 都相同的值。 | Lab 1 §1.3（折叠框） |
| cache line | 缓存和显存之间搬运数据的最小单位，本机 128 字节。 | Lab 1 §1.2 |
| matrix core | CU 里专门做矩阵乘加的硬件单元，通过 MFMA 指令使用。 | Lab 1 §5.5 |
| SPX | MI308X 的一种分区模式：整张卡是一个 HIP 设备，4 个 XCD 共 80 个 CU。 | Lab 0 §2.3 |
| CPX | MI308X 的一种分区模式：每个 XCD 是一个独立的 HIP 设备，各 20 个 CU。 | Lab 0 §2.3 |
| ECC | 纠错码。架构名里的 `sramecc+` 表示片上存储的 ECC 已开启。 | Lab 0 §5.3 |
| xnack | GPU 访存缺页后重试的功能。架构名里的 `xnack-` 表示关闭。 | Lab 0 §5.3 |
| 显存位宽 | HBM 数据总线的总宽度，本机 8192 bit，用来算理论带宽。 | Lab 1 §1.3 |

## 编程模型与 HIP

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| HIP | AMD 的 GPU 编程接口和 C++ 语言扩展，写法与 CUDA 基本一一对应。 | Lab 0 §3 |
| ROCm | AMD 的 GPU 软件栈，包括驱动、hipcc 编译器、各种库（含 CK-Tile）和工具（rocprofv3 等）。 | Lab 0 §1.4 |
| hipcc | ROCm 的 HIP 编译器，路径 `/opt/rocm/bin/hipcc`。 | Lab 0 §1.4 |
| kernel | 在 GPU 上执行的函数。主机代码启动一次 kernel，会有大量线程同时执行同一个函数体。 | Lab 0 §3.1 |
| thread | 执行 kernel 的最小单位，每个线程有自己的下标和寄存器。AMD 文档里也叫 work-item。 | Lab 0 §2.2 |
| block | 一组线程，最多 1024 个。同一个 block 的线程在同一个 CU 上运行，可以通过 LDS 和 `__syncthreads()` 合作。AMD 文档里叫 workgroup。 | Lab 0 §2.2 |
| grid | 一次 kernel 启动的全部 block。 | Lab 0 §2.2 |
| `__global__` | 声明 kernel 的关键字：函数在 GPU 上执行，由主机代码启动。 | Lab 0 §3.1 |
| `<<< >>>` 启动语法 | 启动 kernel 的写法：`kernel<<<grid, block, LDS 字节数, stream>>>(参数)`。 | Lab 0 §3.1 |
| 动态 LDS | 用 `extern __shared__` 声明、大小在启动时由 `<<< >>>` 第 3 个参数指定的 LDS。它叠加在静态 `__shared__` 数组之上；本课不用，第 3 个参数填 0。 | Lab 1 §2.2 |
| `threadIdx.x` | 内建变量：当前线程在 block 里的 x 方向下标。 | Lab 0 §3.2 |
| `blockIdx.x` | 内建变量：当前 block 在 grid 里的 x 方向下标。 | Lab 0 §3.2 |
| `blockDim.x` | 内建变量：每个 block 在 x 方向的线程数。 | Lab 0 §3.2 |
| `gridDim.x` | 内建变量：grid 在 x 方向的 block 数。 | Lab 1 §2.2 |
| `warpSize` | 内建变量：wavefront 宽度，本机 64。不要把 64 或 32 写死在代码里（Lab 0 §5.2 用 `hipDeviceAttributeWarpSize` 查过同一个值）。ROCm 7 里它只能在设备代码里当运行时值用，不能用在需要编译期常量的地方。 | Lab 1 §4.3 |
| stream | 命令队列，类型 `hipStream_t`。同一个 stream 里的操作按顺序执行；传 0 表示默认 stream。 | Lab 0 §3.1 |
| 异步执行 | 启动 kernel 后主机代码不等它结束就继续往下走。需要结果时要显式同步。 | Lab 0 §4.2 |
| `hipMalloc` | 在显存上分配内存。 | Lab 0 §4.1 |
| `hipMemcpy` | 在主机和 GPU 之间、或 GPU 内部拷贝数据。 | Lab 0 §4.2 |
| `hipMemset` | 把一段显存的每个字节都设成同一个值（按字节设置，不是按元素）。 | Lab 0 §6.3 |
| `hipFree` | 释放 `hipMalloc` 分配的显存。 | Lab 0 §4.3 |
| `hipGetLastError` | 取出最近一次 HIP 调用（包括 kernel 启动）留下的错误码。 | Lab 0 §4.2 |
| `hipGetErrorString` | 把错误码转成可读的字符串。 | Lab 0 §4.1 |
| `hipGetDeviceProperties` | 一次查出设备的全部属性，填进 `hipDeviceProp_t` 结构体。 | Lab 0 §5.1 |
| `hipDeviceGetAttribute` | 一次查设备的一项属性，例如 CU 数（`hipDeviceAttributeMultiprocessorCount`）。 | Lab 0 §5.2 |
| `__restrict__` | 编译器扩展：承诺这个指针指向的内存不会通过别的指针访问，编译器可以据此做更多优化。 | Lab 2 §1.1 |
| SIMT | Single Instruction, Multiple Threads。按单个线程写程序，硬件以 wavefront 为单位执行，64 个 lane 执行同一条指令、各用自己的寄存器。 | Lab 1 §2.1 |
| 执行掩码 | EXEC mask，64 位，每位对应一个 lane。被关掉的 lane 不写结果，`if` 条件为假的 lane 就是这样被关掉的。 | Lab 1 §2.1 |
| 分支发散 | 同一个 wavefront 的 lane 走不同分支。两条分支都要依次执行，各自只有一部分 lane 在干活。 | Lab 1 §2.1 |
| 向上取整 | 算 grid 大小的写法 `(n + block − 1) / block`，保证线程总数不少于 n。 | Lab 0 §3.3 |
| grid-stride loop | 每个线程从自己的全局下标出发，每次前进整个 grid 的线程数，直到越过 n。这样 grid 大小可以与 n 无关。 | Lab 1 §2.5 |
| `__shared__` | 在 kernel 里声明 LDS 变量的关键字，同一个 block 的线程看到同一份。 | Lab 1 §3.4 |
| `__syncthreads()` | block 级屏障：block 里所有线程都到达后才继续，并保证之前的 LDS 写入对 block 内其他线程可见。 | Lab 1 §3.5 |
| 写后读 / 读后写 | 两种需要屏障的依赖。写后读：读 LDS 之前，别的线程必须已经写完；读后写：覆盖 LDS 之前，别的线程必须已经读完。 | Lab 1 §5.3 |
| 数据竞争 | 两个线程访问同一地址、至少一个是写，而中间没有同步。结果取决于执行顺序，可能时对时错。 | Lab 1 §3.5 |
| 原子操作 | 对一个地址的“读—改—写”作为不可分割的整体完成，多个线程同时做也不会丢失更新。 | Lab 1 §4.5 |
| `atomicAdd` | 原子加：把一个值加到指定地址上，返回加之前的旧值。多个线程的加法都不会丢，但先后顺序不确定。 | Lab 1 §4.5 |
| `__shfl_down` | wavefront shuffle 函数：每个 lane 取编号比自己大 `lane_delta` 的 lane 的值，源 lane 超出范围时拿回自己的值。不经过 LDS 数组，也不需要屏障。 | Lab 1 §4.3 |
| 归约 | reduction，把很多个数用一种运算（求和、求最大值等）合成一个或少数几个结果。 | Lab 1 §4 |
| 树形归约 | 每一层把数据两两合并、数量减半的归约方式，n 个数需要 log₂n 层。 | Lab 1 §4.2 |
| 行主序 | row-major，矩阵按行连续存放，元素 (i, j) 的偏移是 i × 每行元素数 + j。 | Lab 1 §5.1 |
| scratch | 每个线程私有、放在显存里的存储。寄存器不够（溢出）或局部数组无法放进寄存器时，编译器会用它，访问比寄存器慢得多。 | Lab 1 §1.2，§5.4 详讲 |

## 性能

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| 延迟掩盖 | 一个 wavefront 等显存数据时，SIMD 去执行另一个驻留的 wavefront，让等待的时间不被浪费。 | Lab 1 §1.1 |
| occupancy | 占用率：每个 SIMD 上实际驻留的 wavefront 数，本机最多 8 个。受寄存器、LDS 用量和 block 大小限制。驻留越多越容易掩盖延迟，但不是越高越快。 | Lab 1 §2.4 |
| 合并访存 | coalesced access：同一个 wavefront 的 lane 访问相邻地址，硬件就能把它们合并成少数几次内存事务。例如 64 个连续 float（256 字节）对齐时只涉及 2 条 cache line。 | Lab 1 §3.1 |
| 广播 | 同一个 wavefront 的多个 lane 读同一个地址时，硬件只取一次再分给这些 lane；在 LDS 里不算 bank conflict。 | Lab 1 §3.6 |
| bank conflict | LDS 分成 32 个 bank，每个 bank 每周期只服务一个地址。同一批访问（每 lane 4 字节时是半个 wavefront）里有 k 个 lane 访问同一 bank 的不同地址，就要串行 k 次。 | Lab 1 §3.6 |
| LDS padding | 给 LDS 二维数组每行多开几列（常见是 1 列）不存数据，错开各行起始的 bank，消除按列访问时的 bank conflict。 | Lab 1 §3.6 |
| swizzle | 不多占 LDS、通过对坐标做 XOR 等变换改变数据到 bank 的映射，用来消除 bank conflict。CK-Tile 里对应 xor 变换。 | Lab 3 §1.6 |
| FMA | Fused Multiply-Add，一条指令算 a × b + c，只舍入一次。一次 FMA 计 2 FLOP。 | Lab 1 §1.3 |
| FLOP | 一次浮点运算。GFLOPS = 每秒 10⁹ 次，TFLOPS = 每秒 10¹² 次。 | Lab 1 §1.3 |
| 计算强度 | 每访问 1 字节显存做多少次浮点运算（FLOP/B）。在只看带宽和算力的简单模型里，它决定哪条上限更低，用来指导优化方向；实际瓶颈还要测量。 | Lab 1 §1.4 |
| fp32 向量峰值 | 不用 matrix core 时 fp32 的理论算力：CU 数 × 每 CU 每周期 FLOP 数 × 频率。本机约 37.9 TFLOPS，实测 FMA 吞吐约 28 TFLOPS。 | Lab 1 §1.3 |
| packed 指令 | 一条指令同时处理两组操作数，例如 `v_pk_fma_f32` 一次做两个 fp32 FMA。 | Lab 1 §1.3 |
| 有效带宽 | kernel 逻辑上必须读写的字节数 ÷ 耗时，用来衡量访存类 kernel。 | Lab 1 §3.7 |
| D2D 拷贝带宽 | 同一进程里 `hipMemcpy` 显存到显存拷贝的实测带宽，按（读 + 写）字节 ÷ 时间计算，本机约 2 TB/s。Lab 1 的访存类性能用例以它为基准。 | Lab 1 §1.3 |
| 微基准 | 只测一种操作（如拷贝、FMA、LDS 读取）的小测试程序。它的结果是这个程序达到的性能，不是硬件上限。 | Lab 1 §1.3 |
| 比值 | 本课程性能用例的归一化方式：你的带宽（或 GFLOPS）÷ 同进程实测的拷贝带宽（或 FMA 吞吐）。Lab 1 写成百分比，Lab 2 起直接写比值。 | Lab 1 §1.3 |
| 数据复用 | 把数据搬到更近的存储（LDS、寄存器）后多次使用，减少显存访问。 | Lab 1 §5.2 |
| 分块 | tiling：把大问题切成小块，每块的数据先装进 LDS 或寄存器再反复使用。 | Lab 1 §5.2 |
| 寄存器分块 | 让每个线程负责多个输出元素，在寄存器里复用已读入的数据，进一步提高计算强度。 | Lab 1 §5.4 |
| `__launch_bounds__` | 写在 kernel 前、告诉编译器 block 最多有多少线程的属性，编译器据此估算资源和占用率。 | Lab 1 §5.4（折叠框） |
| 循环展开 | 编译器把循环体按次数复制多份，每份里的循环变量换成常数。`#pragma unroll` 可以明确要求展开。 | Lab 1 §5.4 |
| LLVM IR | 编译器的中间表示，介于源码和 GPU 汇编之间。`--save-temps` 留下的 `.bc` 文件就是它，用 `llvm-dis` 可以转成文本。 | Lab 2 §2.3 |

## 数值

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| fp32 | IEEE 754 单精度浮点（`float`），32 位，有效数字 24 位。0 到 2²⁴ 之间的整数都能精确表示。 | Lab 1 §4.5 |
| fp16 | IEEE 754 半精度浮点，16 位。 | Lab 1 §5.5 |
| bf16 | 16 位浮点，指数位数与 fp32 相同（表示范围相同），有效数字只有 8 位。 | Lab 1 §5.5 |

## 工具与环境

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| rocminfo | 列出 GPU 硬件属性（CU 数、SIMD 数、LDS 大小等）的命令，路径 `/opt/rocm/bin/rocminfo`。 | Lab 0 §2.3 |
| rocm-smi | 查看每张卡的占用、显存、温度、频率的命令；新工具 amd-smi 功能类似。 | Lab 0 §6.1 |
| `HIP_VISIBLE_DEVICES` | 环境变量：只让程序看到列出的 GPU，看到的卡从 0 重新编号。 | Lab 0 §6.2 |
| rocprofv3 | ROCm 的性能分析工具，可以记录每次 kernel 的起止时间并统计耗时。 | Lab 0 §6.3 |
| `HIP_LAUNCH_BLOCKING` | 环境变量：设为 1 时每次启动 kernel 都等它结束才返回，错误会在出错的那次启动处报出来。 | Lab 1 §6 |
| GuardedBuffer | 本课测试框架里的显存缓冲区：数据前后各留一段填 NaN 的哨兵区，用来发现越界读写。 | Lab 0 §1.3 |

## C++ 编译期编程

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| template | C++ 模板：把类型或常数作为参数，编译期为每组参数生成一份代码。 | Lab 2 §2 |
| `constexpr` | 声明变量或函数可以在编译期求值；函数也可以在运行期调用。 | Lab 2 §2.1 |
| `static_assert` | 编译期断言：条件为假时编译报错。 | Lab 2 §2.1 |
| 参数包 | parameter pack，如 `template <index_t... Is>`，表示任意多个模板参数。在 C++20 里只能整体展开，不能按下标取其中一个。 | Lab 2 §2.1 |
| 折叠表达式 | fold expression，C++17 语法，用一个二元运算符展开整个参数包，如 `(Is + ... + 0)`。 | Lab 2 §2.1 |
| `if constexpr` | 编译期条件分支，没选中的分支不会实例化，不同分支可以返回不同类型。 | Lab 2 §2.4 |
| 泛型 lambda | 参数类型写成 `auto` 的 lambda，编译器为每种实参类型生成一个版本。 | Lab 2 §2.6 |
| `decltype` | 取变量或表达式的类型。CK-Tile 里常用它从参数类型中取出编译期常数。 | Lab 2 §2.6 |
| tag dispatch | 用参数的类型（如 `number<I>`）而不是运行期的值来选择调用哪个重载。 | Lab 2 §2.5 |
| `enable_if` | 写在模板参数里的条件：条件不成立时，这个函数版本对编译器来说“不存在”，调用时会报找不到匹配的函数。 | Lab 2 §3.1 |
| 特化 | 为某组特定的模板参数另写一份实现。例如 `static_for<0, N, 1>` 有一个专门的版本，比一般情形更简单。 | Lab 2 §2.3 |

## CK-Tile

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| CK-Tile | AMD Composable Kernel 库里的 tile 编程接口。用编译期类型描述数据怎样分给线程，由库生成访存代码。 | Lab 2 §1 |
| tile | 一块要整体处理的数据子块。CK-Tile 用类型描述 tile 的形状和分布。 | Lab 2 §1.1 |
| `number` | `ck_tile::number<N>`，把编译期整数包装成一个类型（是 `constant<N>` 的别名）。 | Lab 2 §2.1 |
| `sequence` | `ck_tile::sequence<Is...>`，空结构体，只用模板参数携带一串编译期整数。 | Lab 2 §2.1 |
| `array` | `ck_tile::array`，定长数组，主机和设备代码都能用，用 `a[i]` 读写。 | Lab 2 §2.2 |
| `static_for` | 编译期循环：把循环变量以 `number<i>` 传给泛型 lambda，每次迭代展开成独立的代码。 | Lab 2 §2.3 |
| `tuple` | `ck_tile::tuple`，可以装不同类型的值，只能用 `number<I>` 作下标（`t[number<I>{}]` 或 `t.at(number<I>{})`）。 | Lab 2 §2.5 |
| stride | 沿某一维前进一个元素时，内存偏移增加多少个元素。 | Lab 2 §2.2 |
| tensor view | 一块有形状的内存：指针、各维长度和 stride、地址空间。 | Lab 2 §3.1 |
| buffer view | tensor view 里负责真正读写的部分：指针 + 底层存储的跨度。一维连续数组的跨度就是元素个数；带行间距的二维视图跨度是 (rows − 1) × ld + cols。 | Lab 2 §3.1 |
| `address_space_enum` | 标明 tensor view 的数据在哪里：`global` 是显存，`lds` 是 LDS。 | Lab 2 §3.1 |
| `make_naive_tensor_view` | 用长度和 stride 创建 tensor view。可选地承诺向量长度；不写时 CK-Tile 不会自己推导，只能逐元素访问。 | Lab 2 §3.1 |
| buffer 指令 | AMD GPU 的一类显存访问指令，带一个字节范围（range），越界的读返回 0、越界的写被丢弃。检查粒度是 4 字节，小于 4 字节的类型在末尾可能误伤合法元素。 | Lab 2 §3.1 |
| GuaranteedVectorLengths | 创建视图时的编译期承诺：最后一维可以每 N 个元素一组做向量访问。 | Lab 2 §3.1，Lab 3 §3.4 详讲 |
| 向量化访存 | 一条指令搬运多个连续元素，例如一次读 4 个 float（16 字节）。 | Lab 2 §1.1 |
| tile distribution | 编译期描述 tile 的每个元素分给哪个线程、放在线程的哪个位置。 | Lab 2 §3.2 |
| tile distribution encoding | `tile_distribution_encoding`，用 R、H、P、Y 四组维度写出一个 tile distribution。 | Lab 2 §3.2 |
| R 维 | 复制维：沿这些维度的线程持有相同的数据。 | Lab 2 §3.2 |
| H 维 | 把 tile 的每一维拆成几段（例如按 wavefront、lane、每线程元素）的描述。 | Lab 2 §3.2 |
| P 维 | 线程身份对应的维度。2 个 P 维时分别是 warp id 和 lane id；只有 1 个 P 维时只对应 lane id，多 wavefront 的 block 要用 2 个。 | Lab 2 §3.2 |
| Y 维 | 每个线程自己持有的那几段元素，决定它们在线程寄存器数组里的排列。 | Lab 2 §3.2 |
| `make_static_tile_distribution` | 从 `tile_distribution_encoding` 生成 tile distribution 对象。 | Lab 2 §3.4 |
| tile window | tensor view 上的一块矩形区域，由原点和长度确定，可以带一个 tile distribution。 | Lab 2 §3.3 |
| `make_tile_window` | 在 tensor view 上创建 tile window，指定长度、原点和分布。 | Lab 2 §3.3 |
| `multi_index` | `ck_tile::array<index_t, 维数>` 的别名，一个定长的坐标数组，例如 tile window 的原点。 | Lab 2 §3.3 |
| distributed tensor | 一个 tile 分散在所有线程寄存器里的表示，每个线程持有自己的 Y 段。类型是 `static_distributed_tensor`。 | Lab 2 §3.4 |
| thread buffer | distributed tensor 里每个线程自己的那份寄存器数组，长度等于所有 Y 维长度之积。 | Lab 2 §3.4 |
| `load_tile` | 按 tile window 的分布从显存读数据到寄存器，返回 distributed tensor。 | Lab 2 §3.4 |
| `store_tile` | 把 distributed tensor 按 tile window 的分布写回显存。 | Lab 2 §3.4 |
| `sweep_tile` | 对本线程持有的每个元素调用一次 lambda，下标在编译期展开。 | Lab 2 §3.4 |
| `make_static_distributed_tensor` | 按给定分布创建一个 distributed tensor，通常用来装输出；本机 CK-Tile 默认把元素初始化为 0。 | Lab 2 §3.4 |
| ScalarPerVector | CK-Tile 编译期算出的每次访存搬几个元素，由 Y 段的连续长度和视图承诺的向量长度共同决定。 | Lab 2 §3.1（折叠框），§4.2 |
| layout | 一串坐标变换拼成的对象，把逻辑坐标翻译成内存里的线性偏移。 | Lab 3 §1 |
| 上层维度 | 描述符最外层的逻辑坐标维度，例如矩阵的 (i, j)。 | Lab 3 §1.1 |
| 下层维度 | 某个坐标变换输出给下一层的维度。 | Lab 3 §1.1 |
| hidden id | CK-Tile 给变换链里每个维度的统一编号；上层维度和下层维度都是其中的一部分。 | Lab 3 §1.1 |
| tensor adaptor | 只做坐标翻译、不管内存的对象，由一串变换叠成。 | Lab 3 §1.1 |
| tensor descriptor | tensor adaptor 加上元素空间大小和向量长度承诺，是 tensor view 里描述形状的部分。 | Lab 3 §1.1 |
| `calculate_offset` | 把上层坐标依次经过每个变换，算出线性偏移。 | Lab 3 §1.1 |
| `transform_tensor_descriptor` | 在已有描述符上再叠加坐标变换，返回新的描述符。 | Lab 3 §1.1 |
| embed | 把多维坐标变成线性偏移：Σ 坐标 × stride。naive 描述符就是一个 embed。 | Lab 3 §1.1 |
| pass_through | 恒等变换，输入原样输出，用来在变换列表里占住一个维度。 | Lab 3 §1.2 |
| merge | 把相邻几维合成一维，新长度是各维长度之积。 | Lab 3 §1.3 |
| unmerge | 把一维拆成几维，是 merge 的逆变换，常用来把一维下标拆成“块号 + 块内下标”。 | Lab 3 §1.5 |
| `pad_tensor_view` | 把视图的逻辑长度补到 tile 大小的整数倍；补出来的位置读为 0、写被丢弃，代替手写边界检查。 | Lab 3 §2.3 |
| `block_sync_lds` | CK-Tile 的 block 级屏障：等本线程的 LDS 操作完成后做 `s_barrier`，作用相当于 `__syncthreads()`。 | Lab 3 §2.5 |
| `move_tile_window` | 把 tile window 的原点平移一段，例如在 GEMM 里沿 K 方向滑动。 | Lab 2 §3.3 |
| `block_tile_reduce` | 对 distributed tensor 做每个线程内部的归约，只支持 2D tile 归约第 1 维；跨 lane 的合并要再调 `block_tile_reduce_sync`，它不跨 wavefront。 | Lab 3 §3.1 |

## 课程首页预告、后续 Lab 讲解

| 术语 | 解释 | 首次讲解位置 |
|---|---|---|
| mega kernel | 把多个计算和通信阶段融合进一个常驻 GPU 的 kernel，由 kernel 自己调度各阶段。 | 课程首页预告 |
| GEMM | 通用矩阵乘 C = A × B（可带缩放和累加）。 | Lab 1 §5 |
| MFMA | Matrix Fused Multiply-Add，AMD 的矩阵乘加指令，由一个 wavefront 协作完成一个小矩阵乘。 | Lab 1 §5.5（预告），Lab 4 详讲 |
| 软件流水线 | 在计算当前数据的同时预先加载下一轮数据，让访存和计算重叠。 | 课程首页预告 |
| Grouped GEMM | 一次启动处理多个互相独立、尺寸可以不同的 GEMM。 | 课程首页预告 |
| P2P | peer-to-peer，一张 GPU 直接读写另一张 GPU 的显存，不经过主机内存。 | 课程首页预告 |
| IPC | 进程间通信。GPU 语境下指一个进程把自己的显存共享给另一个进程访问。 | 课程首页预告 |
| RCCL | ROCm 的集合通信库，对应 NVIDIA 的 NCCL。 | 课程首页预告 |
| all-to-all | 一种集合通信：每张 GPU 给其他每张 GPU 各发一份不同的数据。 | 课程首页预告 |
| FlashAttention | 一种分块计算注意力、不把完整注意力矩阵写回显存的算法。CK-Tile 的 ops 里有它的实现（fmha）。 | Lab 2 §1.2（仅提及） |
| MoE | Mixture of Experts，模型里每个 token 只交给少数几个“专家”子网络计算的结构。CK-Tile 的 ops 里有融合实现（fused_moe）。 | Lab 2 §1.2（仅提及） |

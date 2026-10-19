# Lab 3 书面问题

> 书面问题不计分，但它们检验的正是本课三个核心概念：布局 = 坐标变换（a1）、二维分布与 LDS 视图（a2）、在分布式 tile 上归约（a3）。题目原文见各 Assignment 的 README；需要的背景见讲义 §1（layout）、§2（二维分布与 LDS）、§3（归约）。引用源码时请写明文件路径和行号（源码在 `/opt/rocm/include/ck_tile/`）。

## a1 layout（a1-layout/README.md）

### 1. `transposed_view` 没有移动任何数据，它和 Lab 1 a2 中的转置有什么本质区别？什么时候“换视图”就够了，什么时候必须真的搬数据？

- 本质区别：
- “换视图”就够的情形：
- 必须真的搬数据的情形（提示：之后谁来读这块数据？它要求什么样的内存顺序？）：

（在此作答）

### 2. 用 `xor_swizzled<8,8>` 的偏移表说明：按列读取时，8 个元素落在多少个不同的“bank”上（假设 8 个 bank，bank = offset % 8）？与 row-major 比较。

先用 `make run A=a1 CASE=xor_swizzled` 或纸笔写出偏移表，再挑一列计算。注意题目假设的是 8 个 bank，讲义小部件 1.1 用的是真实硬件的 32 个 bank，两者结果不必相同。

| 布局 | 第 j 列的 8 个偏移（选一个 j） | 对 8 取模后的 bank | 不同 bank 个数 |
|------|-------------------------------|--------------------|----------------|
| row-major 8×8 |  |  |  |
| xor_swizzled 8×8 |  |  |  |

- 结论：

（在此作答）

### 3. `make_tiled` 得到的 4 维视图如何用来表达“block (I,J) 负责的 tile 中第 (ii,jj) 个元素”？这与 Lab 1 a4 的手写下标计算对应关系是什么？

- 4 维坐标的含义：
- 与 Lab 1 a4 中 `blockIdx` / `threadIdx` / 循环变量的对应：

（在此作答）

## a2 tile-transpose（a2-tile-transpose/README.md）

### 1. 去掉 `pad_tensor_view`（直接用 naive 视图）时，为什么 1×1 的矩阵结果碰巧正确，而 33×31 会错？

分别分析两个尺寸里“伸出矩阵的那部分 tile”的线性偏移落在哪里（讲义 §2.3、图 2.2）。读和写两个方向都要考虑。

- 1×1：
- 33×31：
- 你的实测（可选：临时把 `vin`、`vout` 换成 naive 视图跑 `make run A=a2 CASE=ragged`，记下首个错误，再改回来）：

（在此作答）

### 2. `lds_r` 的 stride 为 `(1, TN+1)`，它和 a1 的 `make_transposed_view` 有什么关系？有什么不同（一个用 stride，一个用变换）？什么情形下只能用变换、不能只改 stride？

- 关系：
- 不同：
- 只能用变换的情形（提示：回看讲义 §1.4–§1.6，哪些变换的偏移不是坐标的线性函数？）：

（在此作答）

### 3. 用 `make run A=a2 CASE=bandwidth` 记下本作业的比值，与你自己 Lab 1 a2 的 LDS 转置比较（Lab 1 §3.7 的参考实现约 0.31–0.35），解释差异。

带宽有噪声：跑 3–5 次，填最好的一次（连同同一次的拷贝带宽与比值）。

| 实现 | 带宽（GB/s，最好一次） | 同次实测拷贝带宽（GB/s） | 比值 |
|------|------------------------|--------------------------|------|
| 本作业（LDS + padding） |  |  |  |
| 你的 Lab 1 a2 LDS 转置（`lab1-hip-basics/` 下 `make run A=a2 CASE=lds_bandwidth`；拷贝带宽借用上一行同时段的数） |  |  |  |
| 可选：去掉 padding |  |  |  |
| 可选：跳过 LDS |  |  |  |

- 从“每个线程一次访问的字节数”“访存宽度与合并”“LDS 的访问方式”三个角度解释差异（你在 Lab 1 a2 里一个线程一次读写几个元素？本作业呢？）：

（在此作答）

## a3 row-sum（a3-row-sum/README.md）

### 1. 为什么 `cols % 4 ≠ 0` 时声明向量长度 4 会出错？

提示：一个从 4 的倍数列开始的向量会不会伸进下一行？pad 判断有效性时看的是向量的哪个元素？在 gfx942 上行首不对齐并不会读错（讲义 §3.4 实测），对齐只是在别的硬件上可能出现的额外问题。

- 出错的机制（讲义 §3.4；引用 `/opt/rocm/include/ck_tile/core/tensor/buffer_view.hpp` 或 `/opt/rocm/include/ck_tile/core/algorithm/coordinate_transform.hpp` 的相关行）：
- 为什么最后一行反而可能是对的：
- 你的实测（可选：临时把分派条件去掉、总是走向量 4，跑 `make run A=a3 CASE=ragged_cols`，记下首个错误）：

（在此作答）

### 2. `block_tile_reduce_sync` 之后，同一行的结果被 R 段上的多个 lane 同时持有。`store_tile` 时，这些 lane 会不会都去写同一个地址？

- 你的设计里，一行的结果被几个 lane 持有：
- R 维在分布的坐标适配器里是哪种变换，它对 X 坐标有没有贡献（`core/tensor/tile_distribution.hpp` 第 285–292 行、`core/algorithm/coordinate_transform.hpp` 的 `replicate`）：
- 追踪写路径（`store_tile.hpp` → `tile_window.hpp` 的 `store()` → …，写出文件:行号），其中有没有按 R 坐标筛选写入者的逻辑：
- 验证实验与结果（给每个 lane 的值加一个与 lane id 有关的标记）：
- 结论：为什么在本作业里无害；为什么 sync 必须在 store 之前：

（在此作答）

### 3. 如果要支持一行跨 warp（例如 cols 很大而 rows 很小），需要额外做什么？（提示：跨 warp 归约要经过 LDS，类似 Lab 1 a3。）

（在此作答）

### 4. 设计说明：用 3–5 句话写下你的 block 形状、每 block 行数、tile 宽度、Y 维划分、向量长度的选择和理由。

| 项目 | 你的选择 | 理由 |
|------|----------|------|
| block 线程数（几个 wavefront） |  |  |
| 每 block 行数 |  |  |
| tile 宽度（列） |  |  |
| 行方向 H 段与认领者 |  |  |
| 列方向 H 段与认领者 |  |  |
| 向量长度与分派条件 |  |  |
| `make run A=a3 CASE=bandwidth_aligned` 的比值 |  |  |

（在此作答）

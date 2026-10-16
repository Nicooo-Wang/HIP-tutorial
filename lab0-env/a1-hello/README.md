# Lab 0 · a1 hello：第一个 kernel

## 你要做什么

编辑 `a1-hello/hello.hip`，让主机函数

```cpp
void launch_fill_ids(int* out, int n, hipStream_t stream);
```

结束后，GPU 显存中的数组满足 `out[i] == i`（对所有 `0 <= i < n`），并且**不写 `out` 范围之外的任何内存**。

需要补全两处：

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a1-1` | kernel `fill_ids_kernel` | 计算全局下标并做边界检查 | §3.2 |
| `a1-2` | 主机函数 `launch_fill_ids` | 计算足够大的 grid（向上取整） | §3.3 |

block 大小已经固定为 256，不需要改。`out` 由测试分配好、已经在 GPU 显存上；`stream` 原样传给启动语句即可（讲义 §3.1）。

## 怎么判分

在 `lab0-env/` 目录下运行：

```bash
make grade A=a1                       # 判分
make run A=a1 CASE=many_blocks        # 只运行一个用例，看原始输出
```

测试先把 `out` 全部填成 `-1`，调用你的 `launch_fill_ids`，再逐个元素比对，最后检查 `out` 前后各 4096 字节的哨兵区是否被改写（讲义 §1.3）。

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `one_block_exact` | 1 | n = 256，恰好一个满 block |
| `single_element` | 1 | n = 1：只有 1 个元素，其余 255 个线程必须什么都不做 |
| `not_multiple_of_block` | 2 | n = 1000 与 n = 257：最后一个 block 不满，既不能漏写，也不能越界写 |
| `many_blocks` | 2 | n = 4194307（= 2²² + 3）：上万个 block，下标必须跨 block 连续 |

读失败信息的方法：“得到 -1”表示那个位置**没有任何线程写过**；“越界写：……哨兵区被改写”表示有线程写到了数组外面。

## Hints

<details><summary>Hint 1（方向）</summary>

每个线程只负责一个元素。先想清楚：第 `b` 个 block 里第 `t` 个线程应该负责哪个元素？然后想：线程总数（grid × 256）什么时候会比 n 多？多出来的线程该怎么办？最后想：grid 至少是多少才能让线程总数不少于 n？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- a1-1：用 `blockIdx.x`、`blockDim.x`、`threadIdx.x` 算出全局下标 i（讲义 §3.2 的公式）；只有当 i 小于 n 时才写 `out[i]`，否则什么都不做。
- a1-2：grid 是 n 除以 block 大小**向上取整**。C++ 整数除法向下取整，要用讲义 §3.3 的“先给分子加上除数减一”的技巧。检查：n = 1 时 grid 应为 1，n = 256 时为 1，n = 257 时为 2。

</details>

## 书面问题（写到 ../answers.md）

1. 若 n = 1000、blockDim = 256，启动了多少个线程？多出来的线程在做什么？
2. 去掉 `if (i < n)` 后，n = 1000 时会发生什么？为什么“结果看起来对”不代表程序正确？（提示：哨兵）

## 动手探索（不判分）

- 用 `rocprofv3 --kernel-trace -- ./build/run-a1` 看 kernel 名字与耗时（先 `make run A=a1` 生成 `build/run-a1`；完整命令与输出说明见讲义 §6.3）。在 `kernel_trace.csv` 里找到 `many_blocks` 那次启动，它的 `Grid_Size_X` 是多少？和你算的 grid 是什么关系？
- 临时把 block 大小从 256 改成 2048，运行 `make run A=a1`，看看报什么错（讲义 §4.2）。试完记得改回来。

# Lab 1 · a3：归约（LDS 树形 → wavefront shuffle）

## 你要做什么

编辑 `a3-reduction/reduction.hip`，实现 `*out = sum(in[0..n))`（所有指针都在 GPU 显存上）：

```cpp
constexpr int kWorkspaceFloats = 4096;  // 测试提供的工作区大小（float 个数）
void reduce_sum_lds (const float* in, int n, float* out, float* ws, hipStream_t stream);
void reduce_sum_shfl(const float* in, int n, float* out, float* ws, hipStream_t stream);
```

`ws` 是 `kWorkspaceFloats` 个 float 的设备工作区，**内容未初始化**；`out` 里的初值也不可信（测试会预先填成 NaN）。

两遍结构和两个 launch 函数已经给出：

```
in --(多 block，每个 block 一个部分和)--> ws[0..blocks) --(1 个 block 汇总)--> out
```

你需要填四处：

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a3-1` | `partial_sum_kernel` | grid-stride loop：每个线程累加自己负责的元素 | §4.1（复习 a1 Part B） |
| `a3-2` | `block_reduce_lds` | 用 `__shared__` 数组做树形归约，思考每一层之后需要什么 | §4.2 |
| `a3-3` | `wave_reduce` | 用 `__shfl_down` 在一个 wavefront 内归约（wave64，用 `warpSize`） | §4.3 |
| `a3-4` | `block_reduce_shfl` | wave 内归约 → 写入 LDS → 第 0 个 wave 再归约 | §4.4 |

注意：不要写 `out`、`ws` 之外的内存，也不要修改输入；不要假设 `ws` 或 `out` 里已有任何值。

## 怎么判分

```bash
make grade A=a3                          # 判分
make run A=a3 CASE=shfl_bandwidth        # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `lds_exact_ones` | 3 | LDS 版本，全 1 输入，n = 1、63、64、65、257、4097、2^20+1 等，**精确**比较 |
| `lds_random` | 2 | LDS 版本，随机输入，相对容差比较 |
| `shfl_exact_ones` | 3 | shuffle 版本，同上 |
| `shfl_random` | 2 | shuffle 版本，同上 |
| `repeatable_output` | 1 | 连续调用两次，结果必须一致 |
| `shfl_bandwidth`（性能） | 3 | n = 2^26 的读带宽须达到同进程实测拷贝带宽的 3%（参考实现约 75%；只用 1 个 block 的实现只有约 0.2%） |

全 1 输入在 n < 2^24 时，任何求和顺序的结果都精确等于 n，所以可以精确比较，漏掉或重复一个元素都会被抓住。

## Hints

<details><summary>Hint 1（方向）</summary>

- 为什么需要两遍 kernel？想想 block 之间能不能用 `__syncthreads()` 互相等待。
- 每个线程先用 grid-stride loop 把自己负责的元素累加到一个寄存器里，再做 block 内归约；这样 block 数可以固定，与 n 无关。
- 树形归约每一层活跃线程减半：第一层 `stride = kBlock/2`，线程 `t < stride` 把 `s[t + stride]` 加到 `s[t]`。
- shuffle 让同一 wavefront 的线程不经过 LDS，直接交换寄存器里的值。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- LDS 版：先 `s[tid] = v`，同步；然后每一层“读 → 加 → 同步”，同步不能放进 `if` 里面（所有线程必须都到达）。
- `wave_reduce`：`offset` 从 `warpSize / 2` 开始每次减半，`v += __shfl_down(v, offset)`；循环结束后 lane 0 持有整个 wavefront 的和。
- `block_reduce_shfl`：一个 block 有 `kBlock / warpSize` 个 wave；每个 wave 的 lane 0 把结果写进一个很小的 `__shared__` 数组，同步后让第 0 个 wave 读这些部分和（不存在的 lane 补 0），再做一次 `wave_reduce`。
- 第 2 遍的输入只有 `blocks` 个元素，grid-stride loop 同样适用。

</details>

## 书面问题（写到 ../answers.md）

1. 把 `wave_reduce` 的 offset 从 16 开始（写死 32 线程的 wavefront）在 NVIDIA GPU 上是对的，在 MI308X 上为什么错？
2. 为什么随机输入只能用容差比较，而全 1 输入可以精确比较？对 2^25 个 1 求和时，单线程顺序累加的结果是多少？为什么？
3. 第 2 遍只用 1 个 block，会成为瓶颈吗？用 `rocprofv3 --kernel-trace` 看两个 kernel 各自耗时。
4. （思考）能否只用一个 kernel 完成归约？（提示：atomic，或“最后一个完成的 block 负责汇总”——这是后面 mega kernel 中常见的跨 block 同步手法的雏形。）

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
| `lds_exact_ones` | 3 | LDS 版本，全 1 输入，n = 1、63、64、65、257、4097、2^20+1 等，以及 2^22+3、2^23+1、2^24−1 这样的大 n，**精确**比较；整组重复 3 遍 |
| `lds_random` | 2 | LDS 版本，随机输入，相对容差比较 |
| `shfl_exact_ones` | 3 | shuffle 版本，同上 |
| `shfl_random` | 2 | shuffle 版本，同上 |
| `repeatable_output` | 1 | 连续调用两次，结果必须一致 |
| `shfl_bandwidth`（性能） | 3 | n = 2^26 − 1 的读带宽须达到同进程实测拷贝带宽的 28%；输入是随机的 0/1，结果须精确等于 1 的个数 |

全 1 输入在 n < 2^24 时，任何求和顺序的结果都精确等于 n，所以可以精确比较，漏掉或重复一个元素都会被抓住。

性能参考（空闲 MI308X 实测，相对同进程拷贝带宽）：参考实现（简单设计，即骨架的两遍结构）约 67%；把 `kMaxBlocks` 降到 256 也有约 40%，同样能过；更好的设计可以达到甚至超过拷贝带宽本身（归约只读不写）；第 1 遍只用 1 个 block 的实现不到 0.2%。

读失败信息：“得到 nan”说明 `out` 没被写过、或者某个部分和没被写过（`ws`、`out` 初始都填 NaN），也可能是读越界——`in` 前后的哨兵区填的也是 NaN。

## Hints

<details><summary>Hint 1（方向）</summary>

- 为什么需要两遍 kernel？想想 block 之间能不能用 `__syncthreads()` 互相等待。
- 每个线程先用 grid-stride loop 把自己负责的元素累加到一个寄存器里，再做 block 内归约；这样 block 数可以固定，与 n 无关。
- 树形归约每一层活跃线程减半。用讲义小部件 4.1 的“顺序寻址”模式数一数：第一层哪些线程活跃、各自加的是哪个槽位？之后每层怎么变？
- shuffle 让同一 wavefront 的线程不经过 LDS，直接交换寄存器里的值。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- LDS 版：每个线程的值什么时候进 LDS？读别的线程写的槽位之前要保证什么？这个保证放在哪里才能让 block 里**每个**线程都执行到（包括这一层不活跃的线程）？
- `wave_reduce`：在讲义图 4.2 的 8 lane 玩具上用 a…h 推演：第一次 `__shfl_down` 的偏移取多少，才能让 lane 0..3 各自拿到一个“还没被加过”的值？之后每次偏移怎么变、一共要几次？把 8 换成 `warpSize` 再写代码。
- `block_reduce_shfl`：一个 block 有几个 wave？每个 wave 的部分和放在哪、谁来读？第 0 个 wave 里没有对应部分和的 lane 该拿什么值（讲义 §4.4）？
- 第 2 遍的输入只有 `blocks` 个元素，你的 grid-stride loop 不用改也应该对——想想为什么。

</details>

## 判分覆盖不到的 bug

① 树形归约里漏掉某一层的 `__syncthreads()` 是数据竞争：本机上精确用例（整组重复 3 遍）通常能抓到，但不保证每次都抓到。② `acc` 等累加变量忘了初始化是未定义行为，编译器可能恰好给出 0。“全部 OK”不能证明没有这两类问题——对每一层问自己讲义 §4.2 的两个问题。

## 书面问题（写到 ../answers.md）

1. 把 `wave_reduce` 的 offset 从 16 开始（写死 32 线程的 wavefront）在 NVIDIA GPU 上是对的，在 MI308X 上为什么错？
2. 为什么随机输入只能用容差比较，而全 1 输入可以精确比较？对 2^25 个 1 求和时，单线程顺序累加的结果是多少？为什么？
3. 第 2 遍只用 1 个 block，会成为瓶颈吗？用 rocprofv3 看两个 kernel 各自耗时：
   ```bash
   make run A=a3 CASE=shfl_bandwidth
   rocprofv3 --kernel-trace --stats --output-format csv -d prof -o a3 -- ./build/run-a3 shfl_bandwidth
   ```
   在 `prof/a3_kernel_trace.csv` 里按 `Grid_Size_X`（线程总数）区分两遍（讲义 §6）。
4. （思考，属于拓展；`atomicAdd` 的语义见讲义 §4.5 末尾）能否只用一个 kernel 完成归约？（提示：atomic，或“最后一个完成的 block 负责汇总”——这是后面 mega kernel 中常见的跨 block 同步手法的雏形。）

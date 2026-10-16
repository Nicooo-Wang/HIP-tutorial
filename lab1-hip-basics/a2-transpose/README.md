# Lab 1 · a2：矩阵转置（naive → LDS 分块）

## 你要做什么

编辑 `a2-transpose/transpose.hip`。输入 `in` 是 `rows × cols` 的行主序矩阵，输出 `out` 是 `cols × rows` 的行主序矩阵，要求 `out[c*rows + r] = in[r*cols + c]`（两个指针都在 GPU 显存上）：

```cpp
void launch_transpose_naive(const float* in, float* out, int rows, int cols, hipStream_t stream);
void launch_transpose_lds(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a2-1` | `transpose_naive_kernel` | 2D 线程坐标 → (r, c)，边界检查，写 out | §3.2 |
| `a2-2` | `transpose_lds_kernel` | 设计 LDS 分块转置（tile、block 形状、padding、同步） | §3.4–3.6 |
| `a2-3` | `launch_transpose_lds` | 换成你的 kernel 与 grid/block | §3.4 |

注意：不要写数组范围之外的内存，也不要修改输入。骨架中 `launch_transpose_lds` 暂时调用 naive 版本，需要你换掉。

## 怎么判分

```bash
make grade A=a2                          # 判分
make run A=a2 CASE=lds_bandwidth         # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `naive_square` | 1 | 32×32 方阵 |
| `naive_all_shapes` | 2 | 1×1、33×31、1000×17、17×1000、1024×2048 等：非方阵、非整除 |
| `lds_square` | 1 | LDS 版本 64×64 方阵 |
| `lds_all_shapes` | 3 | LDS 版本同上全部形状；越界读写会被哨兵检测到 |
| `lds_bandwidth`（性能） | 3 | 8192×8192：有效带宽（读 + 写）须达到同进程实测 D2D 拷贝带宽的 14% |

性能用例以同一次运行中测得的拷贝带宽为基准，所以机器上别人在跑任务时也相对公平；失败信息里会给出实测带宽。性能用例在计时之后还会在 CPU 上逐元素校验整个 8192×8192 输出。空闲 MI308X 上实测：naive 约为拷贝带宽的 6%，必须用 LDS 才能通过；参考实现（一个简单的 LDS 分块设计）约 31–35%；更好的设计可以更高（本机测到过约 55–60%）。

读失败信息：“得到 nan”通常表示该元素没有线程写过（`out` 初始填的是 NaN），也可能是读越界——`in` 前后的哨兵区填的也是 NaN；“越界写”表示写到了 `out` 之外。

> **判分覆盖不到的 bug。** 漏掉 `__syncthreads()` 属于数据竞争：结果取决于各 wavefront 的相对快慢。本机上它通常会让 LDS 用例失败，但这是概率性的——某次运行“碰巧”全部通过，不代表代码没有竞争。每次读 LDS 之前，自己确认“写它的线程一定已经写完了”（讲义 §3.5）。

## Hints

<details><summary>Hint 1（方向）</summary>

- 画出 naive 版本里一个 wavefront（64 个线程）读 `in`、写 `out` 的地址，哪一个是连续的？另一个相邻线程间隔多远？
- LDS 可以当作“中转站”：先把一小块矩阵按连续方向读进来，再按转置后的方向取出来写出去，这样读和写都能连续。
- LDS 有 32 个 bank，每个 4 字节宽，同一时刻访问同一 bank 的不同地址会串行化。想想按列读一个 64 列宽的数组会发生什么。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- 一个 wavefront 是 `threadIdx.y` 相同的一行线程（讲义 §3.2）。要让它读 `in` 时地址连续，block 的 x 维应该取多大？tile 比 block 大时，一个线程要负责几个元素、按什么跨度循环（讲义 §3.4）？
- 读入阶段，wavefront 的 64 个 lane 对应 `in` 的哪一行、哪些列？它们各自写到 LDS 数组的哪个位置？
- 写出阶段，要让 wavefront 写 `out` 时地址连续，64 个 lane 应该对应 `out` 中的什么？这些值在 LDS 数组里排成一行还是一列？用讲义 §3.6 的公式算一算这时的 bank。
- 写出的目标是 `out`：它的行数和列数是多少？边界检查该用哪个？

</details>

## 书面问题（写到 ../answers.md）

1. naive 版本中读和写哪一个是非合并访存？为什么？
2. 去掉 `+1` padding（如果你加了）后结果还正确吗？性能会怎样？用 `make run A=a2 CASE=lds_bandwidth` 实测并解释。（在本机上，有的设计去掉 padding 后带宽差别不大，甚至略高——这不说明你测错了。如实记录你的数字，并想想此时整个 kernel 的瓶颈在哪里。）
3. 去掉 `__syncthreads()` 会出什么问题？为什么这种 bug 可能“偶尔对”？（这也是判分不能保证每次都抓到这种 bug 的原因。）

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

性能用例以同一次运行中测得的拷贝带宽为基准，所以机器上别人在跑任务时也相对公平；失败信息里会给出实测带宽。naive 版本的带宽远低于此阈值，必须用 LDS 才能通过。

## Hints

<details><summary>Hint 1（方向）</summary>

- 画出 naive 版本里一个 wavefront（64 个线程）读 `in`、写 `out` 的地址，哪一个是连续的？另一个相邻线程间隔多远？
- LDS 可以当作“中转站”：先把一小块矩阵按连续方向读进来，再按转置后的方向取出来写出去，这样读和写都能连续。
- LDS 有 32 个 bank，每个 4 字节宽，同一时刻访问同一 bank 的不同地址会串行化。想想按列读一个 64 列宽的数组会发生什么。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- 一个 block 负责一个 tile×tile 的子块；block 的 x 维取 64，让一个 wavefront 读一整行；block 的 y 维可以更小，每个线程循环处理多行。
- 读入：线程 (x, y) 把 `in` 的 (r0+y, c0+x) 放进 `tile[y][x]`，注意越界检查。
- 同步之后：输出矩阵的行号 = 输入子块的列号。让 x 维线程对应输出的连续列（即输入的连续行），从 `tile[x][y]` 读出。
- 输出的边界条件和输入的不同：行数、列数互换了，别写反。

</details>

## 书面问题（写到 ../answers.md）

1. naive 版本中读和写哪一个是非合并访存？为什么？
2. 去掉 `+1` padding（如果你加了）后结果还正确吗？性能会怎样？用 `make run A=a2 CASE=lds_bandwidth` 实测并解释。
3. 去掉 `__syncthreads()` 会出什么问题？为什么这种 bug 可能“偶尔对”？（这也是判分没有把它当作必挂 mutant 的原因。）

# Lab 1 · a1：向量加法与 grid-stride loop

## 你要做什么

编辑 `a1-vector-add/vector_add.hip`，完成两个主机函数（`a`、`b`、`c`、`x`、`y` 都已经是 GPU 显存上的指针，`stream` 原样传给启动语句）：

```cpp
// Part A：c[i] = a[i] + b[i]，grid 由你决定
void launch_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream);
// Part B：y[i] = alpha * x[i] + y[i]，grid 固定为 kSaxpyGrid(=160) 个 block × 256 线程
void launch_saxpy(float alpha, const float* x, float* y, int n, hipStream_t stream);
```

要求不写数组范围之外的内存，也不修改输入。

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a1-1` | `vector_add_kernel` | 全局下标 + 边界检查 + 加法 | §2.3 |
| `a1-2` | `launch_vector_add` | 选 block 大小，向上取整算 grid | §2.4 |
| `a1-3` | `saxpy_kernel` | grid-stride loop | §2.5 |

注意：Part B 的 grid 是固定的（`kSaxpyGrid` 与启动语句**不要改**），当 n 大于线程总数时，只靠“一个线程一个元素”覆盖不了，必须让线程循环。

## 怎么判分

在 `lab1-hip-basics/` 目录下运行：

```bash
make grade A=a1                       # 判分
make run A=a1 CASE=saxpy_needs_stride # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `add_small` | 1 | n = 1 与 n = 7：比一个 block 还少 |
| `add_block_exact` | 1 | n = 1024：恰好整除 |
| `add_ragged` | 2 | n = 1023 与 100003：不整除，最后一个 block 不满，不能越界 |
| `add_large` | 2 | n = 2²⁴ + 3：大数组，且结果必须与 CPU 的 float 加法数值相等（零容差） |
| `saxpy_fits_grid` | 1 | n = 160 × 256：恰好一线程一元素 |
| `saxpy_small` | 1 | n = 5：线程远多于元素 |
| `saxpy_needs_stride` | 3 | n = 160 × 256 × 37 + 11：每个线程要处理 37~38 个元素 |

三个 saxpy 用例都依次用 alpha = 1.75、−0.5、0 各跑一遍，并检查输入 x 没有被修改。

读失败信息：“得到 nan”通常表示该元素没有任何线程写过（c 初始填的是 NaN）；也可能是读越界——数组外的哨兵区填的也是 NaN，越界读进来参与加法，结果就是 NaN。“越界写”表示写到了数组外的哨兵区。

## Hints

<details><summary>Hint 1（方向）</summary>

- Part A：为什么 grid 必须**向上取整**？想想 n = 1023、block = 256 时，向下取整会漏掉哪些元素；多出来的线程又该怎么处理？
- Part B：一个线程处理完第 i 个元素后，下一个该处理的元素是哪个？如果整个网格里所有线程“并排”向前走一步，一步要走多远？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- a1-1：和 Lab 0 a1 的全局下标是同一个问题。这次多了两个输入数组——哪些线程连**读**都不该读？
- a1-2：block 大小有哪几条约束（讲义 §2.4 的三条规则）？grid 的向上取整你在 Lab 0 已经推过一次；用 n = 1、256、257 验算你的表达式。
- a1-3：一个线程从哪个元素开始？处理完一个元素后，它和网格里其他所有线程“并排”前进一步，这一步跨过的元素个数等于什么——在 kernel 里用哪两个内置变量能算出来（讲义 §2.5）？循环什么时候停？

</details>

## 书面问题（写到 ../answers.md）

1. Part B 里 n = 1 时，有多少线程什么都没做？这样的浪费在什么情况下值得？
2. grid-stride loop 相比“线程数 = n”的写法有什么好处？（提示：launch 开销、复用线程、可调占用）
3. 用 `make run A=a1 CASE=add_large` 生成 `build/run-a1`，再用 rocprofv3 查看 `add_large` 中 vector_add kernel 的时间：
   ```bash
   rocprofv3 --kernel-trace --stats --output-format csv -d prof -o a1 -- ./build/run-a1 add_large
   ```
   （`--output-format csv` 不能省，否则只生成一个 `.db` 数据库文件。）在 `prof/a1_kernel_trace.csv` 里找到 `vector_add_kernel` 那一行，用 `End_Timestamp - Start_Timestamp`（纳秒）算时间，带宽 = 3 × 4 × n 字节 / 时间，与 MI308X HBM 峰值（讲义 §1 表格）比较。差距有多大？可能的原因是什么？

## 动手探索（不判分）

- 把 `launch_vector_add` 的 block 依次改成 64、128、256、1024，用上面的 rocprofv3 命令比较 `add_large` 的 kernel 时间，有区别吗？
- 把 `kSaxpyGrid` 临时改成 1、10、1000（测试里随之变化），saxpy 的结果还对吗？说明 grid-stride loop 与 grid 大小解耦。试完记得改回 160。

# Lab 1 书面问题

> 书面问题不计分，但都是后面几课要用到的直觉。题目原文见各 Assignment 的 README；需要的硬件数字见讲义 §1.3 的表格。

## a1 vector-add（a1-vector-add/README.md）

### 1. Part B 里 n = 1 时，有多少线程什么都没做？这样的浪费在什么情况下值得？

（在此作答）

### 2. grid-stride loop 相比“线程数 = n”的写法有什么好处？（提示：launch 开销、复用线程、可调占用）

（在此作答）

### 3. 用 rocprofv3 测出 `add_large` 中 vector_add kernel 的时间，算出带宽（3 × 4 × n 字节 / 时间），与 MI308X HBM 峰值（讲义 §1 表格）比较。差距有多大？可能的原因是什么？

命令：

```bash
make run A=a1 CASE=add_large
rocprofv3 --kernel-trace --stats --output-format csv -d prof -o a1 -- ./build/run-a1 add_large
```

- kernel 时间（ns）：
- 带宽（GB/s）：
- 与 HBM 峰值之比：
- 原因分析：

（在此作答）

## a2 transpose（a2-transpose/README.md）

### 1. naive 版本中读和写哪一个是非合并访存？为什么？

（在此作答）

### 2. 去掉 `+1` padding（如果你加了）后结果还正确吗？性能会怎样？用 `make run A=a2 CASE=lds_bandwidth` 实测并解释。

- 有 padding 时带宽：
- 无 padding 时带宽：
- 解释：

（在此作答）

### 3. 去掉 `__syncthreads()` 会出什么问题？为什么这种 bug 可能“偶尔对”？

（在此作答）

## a3 reduction（a3-reduction/README.md）

### 1. 把 `wave_reduce` 的 offset 从 16 开始（写死 32 线程的 wavefront）在 NVIDIA GPU 上是对的，在 MI308X 上为什么错？

（在此作答）

### 2. 为什么随机输入只能用容差比较，而全 1 输入可以精确比较？对 2^25 个 1 求和时，单线程顺序累加的结果是多少？为什么？

（在此作答）

### 3. 第 2 遍只用 1 个 block，会成为瓶颈吗？用 `rocprofv3 --kernel-trace` 看两个 kernel 各自耗时。

- 第 1 遍耗时：
- 第 2 遍耗时：
- 结论：

（在此作答）

### 4. （思考）能否只用一个 kernel 完成归约？（提示：atomic，或“最后一个完成的 block 负责汇总”。）

（在此作答）

## a4 gemm（a4-gemm/README.md）

### 1. 计算 naive 与 tiled（TILE=32）两种实现每计算一次 FMA 需要的全局内存读取字节数（计算强度）。

- naive：
- tiled（TILE=32）：

（在此作答）

### 2. 用你的 GFLOPS 与 MI308X fp32 向量峰值（讲义 §1 表格）比较，差距主要来自哪里？

- 你的 tiled_gflops 结果：
- 占峰值的比例：
- 差距来源：

（在此作答）

### 3. 设计说明：用 3–5 句话写下你的 tile 尺寸、block 形状、每线程输出数、LDS 摆放等选择和理由。

（在此作答）

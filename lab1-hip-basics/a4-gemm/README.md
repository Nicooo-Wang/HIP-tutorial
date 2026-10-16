# Lab 1 · a4：GEMM（naive → LDS 分块）

## 你要做什么

编辑 `a4-gemm/gemm.hip`，实现 `C[M×N] = A[M×K] · B[K×N]`（全部行主序 fp32，指针都在 GPU 显存上）：

```cpp
void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream);
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream);
```

| TODO | 位置 | 内容 | 难度 | 讲义 |
|------|------|------|------|------|
| `a4-1` | `gemm_naive_kernel` | 每个线程计算 C 的一个元素（launch 函数已给出） | 1 | §5.1 |
| `a4-2` | `launch_gemm_tiled` | **自主设计** LDS 分块 GEMM：只给接口，其余自己定 | 3 | §5.2–5.4 |

tiled 部分你需要自己决定：tile 尺寸、block 形状、每个线程算几个输出、LDS 怎么摆放、K 方向怎么循环、边界怎么处理。可以复用 a2 学到的 padding 和 a3 学到的同步。

注意：不要写 `C` 之外的内存，也不要修改输入；`C` 的初值不可信（测试会预先填成 NaN）；M、N、K 都可能不是 tile 的整数倍，K 也可能是 1。

## 怎么判分

```bash
make grade A=a4                          # 判分
make run A=a4 CASE=tiled_gflops          # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `naive_small` | 1 | naive，4×5×6 |
| `naive_all_shapes` | 3 | naive，(M,N,K) = 1×1×1、17×33×65、64³、128×96×200、500×300×257、512³ |
| `tiled_aligned` | 2 | tiled，128³ |
| `tiled_all_shapes` | 4 | tiled，同上全部形状（含非整除、K 不是 tile 的倍数） |
| `tiled_gflops`（性能） | 4 | M=N=K=2048 的 fp32 GEMM 须达到 2000 GFLOPS（空闲 MI308X 实测：参考实现约 4600；naive 约 1150） |

正确性按逐元素误差界判定：`|C - ref| ≤ 1e-5 · Σ|a·b| + 1e-6`，与求和顺序无关，所以你可以随意安排 K 方向的累加顺序。

## Hints

<details><summary>Hint 1（方向）</summary>

- naive 中每个 A、B 元素被从全局内存读了多少次？有没有办法让 block 内的线程共享这些读取？
- 把 A 的一个行块和 B 的一个列块搬到 LDS，block 内所有线程复用；K 方向分段循环。
- 让相邻线程（threadIdx.x 相邻）访问相邻的全局地址，才能合并访存。

</details>

<details><summary>Hint 2（关键思路）</summary>

- 分块的核心：block 负责 C 的一个子块，沿 K 方向分段循环；每一段把 A、B 对应的子块协作搬进 LDS，再从 LDS 里做乘加。
- 每一段需要两次 `__syncthreads()`：一次保证搬运完成，一次保证下一段覆盖 LDS 前都已用完。
- 越界的元素在加载时补 0，K 方向的最后一段同样如此。

</details>

<details><summary>Hint 3（步骤提纲）</summary>

- 每一段（tile）的流程：协作加载 A、B 子块到 LDS → `__syncthreads()` → 用 LDS 数据做 TILE 次乘加 → `__syncthreads()`。
- 补 0 之后乘加循环不需要边界判断；边界判断只放在加载和最终写 C 两处；`__syncthreads()` 不能放进 `if` 里。
- 每个线程只算 1 个输出时，LDS 读与乘加之比较高；让每个线程多算几个输出（寄存器里放多个累加器）可以进一步提速。

</details>

## 书面问题（写到 ../answers.md）

1. 计算 naive 与 tiled（TILE=32）两种实现每计算一次 FMA 需要的全局内存读取字节数（计算强度）。
2. 用你的 GFLOPS 与 MI308X fp32 向量峰值（讲义 §1 表格）比较，差距主要来自哪里？（铺垫 Lab 4：MFMA 矩阵核心）
3. 设计说明（写在 answers.md）：用 3–5 句话写下你的 tile 尺寸、block 形状、每线程输出数、LDS 摆放等选择和理由。

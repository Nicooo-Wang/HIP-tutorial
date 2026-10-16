# Lab 3 · a2：用 CK-Tile 做 LDS 转置

## 你要做什么

编辑 `a2-tile-transpose/tile_transpose.hip`。输入 `in` 是 `rows × cols` 的行主序矩阵，输出 `out` 是 `cols × rows` 的行主序矩阵，要求 `out[c*rows + r] = in[r*cols + c]`：

```cpp
void launch_ck_transpose(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

这与 Lab 1 a2 是同一件事。那时你用 `threadIdx` 和下标公式手写；这次用 CK-Tile 把“谁读哪个元素、数据怎么摆”都*描述*出来：

```
global 视图(pad) → 2D tile distribution → load_tile → store_tile 到 LDS(写入视图)
  → block_sync_lds → load_tile 自 LDS(转置读出视图) → store_tile 到 global
```

常量 `TM/TN`（每个 block 一个 64×64 的 tile，256 线程）、`smem` 声明、`vin`（含 `pad_tensor_view`，作为示例）和 `launch_ck_transpose` 已给出，`stream` 原样传给启动语句，不要改。

| TODO | 内容 | 讲义 |
|------|------|------|
| `a2-1` | 仿照 `vin` 构造 `vout`（`out` 的形状是 cols×rows） | §2.3 |
| `a2-2` | 设计 64×64 tile 的 distribution（H/P/Y 划分 → encoding） | §2.1–2.2 |
| `a2-3` | 为同一块 `smem` 构造两个 LDS 视图：写入视图与转置读出视图 | §2.4 |
| `a2-4` | global → 寄存器 → LDS → 同步 → LDS（转置视图）→ 寄存器 → global | §2.5 |

注意：

- 骨架里除 `vin` 外没有别的语句，原样提交时所有用例都 FAIL（输出保持初始的 NaN），这是预期的。
- 你不需要写任何 `if (r < rows && c < cols)`：边界由 `pad_tensor_view` 负责，书面问题 1 会让你分析它为什么必不可少。
- 不要写数组范围之外的内存，也不要修改输入。

## 怎么判分

在 `lab3-cktile-concepts/` 目录下运行：

```bash
make grade A=a2                          # 判分
make run A=a2 CASE=ragged                # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `one_tile` | 2 | 64×64：恰好一个 tile，用上 block 里全部 256 个线程 |
| `multi_tile` | 2 | 128×256 与 256×128：多个 block，输入、输出窗口的原点都要随 block 移动 |
| `ragged` | 4 | 33×31、100×70、1×1、1000×17、17×1000：tile 伸出矩阵边界 |
| `large` | 1 | 2048×1024 |
| `bandwidth`（性能） | 3 | 8192×8192：有效带宽（读 + 写）不低于同进程实测 D2D 拷贝带宽的 19%（只是“不至于慢得离谱”的下限） |

所有正确性用例逐位比较结果，检查 `out` 两侧的哨兵区没有被写、输入没有被修改。“得到 nan”表示该元素没有任何线程写过；整块内容“位置对但没有转置”和“整块位置错”是两种不同的症状，用 `make run` 看首个错误的位置，反推是哪一步出了问题。性能用例通过时会打印本次实测拷贝带宽和你的比值（书面问题 3 要用）。

## Hints

<details><summary>Hint 1（方向）</summary>

- 先在纸上回答四个问题，再动手写 encoding：一个线程一次读同一行的几个连续元素？64 列的一行需要几个线程？256 个线程一次能覆盖几行？要重复几次才能覆盖 64 行？
- 同一 wavefront 的 64 个 lane 要落在同一行还是几行？对全局内存的合并访问意味着什么？
- 两个 LDS 视图指向同一块内存。第二个视图的“行”是第一个视图的什么？想想 a1 里转置视图改变了哪些量。
- LDS 视图的形状和读出窗口的形状各是多少？窗口原点是多少？输出窗口的原点呢，和输入窗口相比哪个坐标先、哪个后？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- 本题的 tile 是 2D 的：行方向 X0、列方向 X1 各自拆成若干段 H。lane id 要同时参与行和列的划分，所以 P 的第二个维度要认领两个 H 段（P→X 与 P→H 里对应位置各写两个数）；P 的第一个维度（warp）只认领一段。
- Y（线程内重复和向量）要分别落在行方向的哪一段、列方向的哪一段？“向量”应该落在哪个方向，为什么（想想内存里谁是连续的）？
- 验证你的划分：每个 H 段恰好被一个 P 或 Y 认领一次；线程总数 = 256；覆盖元素总数 = 64×64。
- a2-3：写入视图与 `vin` 的形状一致；读出视图的形状是转置后的（TN, TM）。两个视图都要考虑每行多留 1 个 float 的 padding（`smem` 已按这个大小声明），读出视图的两个 stride 分别是多少？
- a2-4：LDS 写完之后、读之前需要做什么？它同步的是哪个范围？

</details>

## 判分覆盖不到的 bug

判分只看结果与带宽，不检查你的设计本身：

- **漏掉 `block_sync_lds()`** 是数据竞争，结果取决于各 wavefront 的快慢。本机上它不一定必挂，某次运行碰巧全过不代表代码没有竞争（同 Lab 1 a2）。
- **LDS 没有 padding**（`stride = TN`）仍然能得到正确结果，只是按列读时 bank 冲突更严重。带宽用例的阈值不一定能发现这个问题，请自己对比有无 padding 的带宽（见“动手探索”）。
- **P 维写得不对但碰巧覆盖**：例如只有 1 个 P 维、block 只有 64 线程，结果同样正确，只是每个 block 只有一个 wavefront 在工作（Lab 2 a2 的坑）。判分不检查这点。
- 边缘处理方式不是唯一的：用别的办法保证不越界、不读写错行，只要结果对也能通过，但请想想它与 `pad_tensor_view` 相比的代价。
- **跳过 LDS**：load 之后直接 store 到一个 stride 为 (1, rows) 的输出视图（跨步写全局内存），结果同样正确，也能过带宽下限（约 0.28–0.29 倍拷贝带宽，而 LDS 设计约 0.54）。判分不检查你是否用了 LDS，请自己动手比较。
- 带宽用例只在 8192×8192 一种规模下测。

## 书面问题（写到 ../answers.md）

1. 去掉 `pad_tensor_view`（直接用 naive 视图）时，为什么 1×1 的矩阵结果碰巧正确，而 33×31 会错？（提示：buffer 指令只检查线性偏移是否越过整个数组的末尾，不检查行、列各自是否越界。分别分析 1×1 和 33×31 里“伸出矩阵的那部分 tile”的线性偏移落在哪里。）
2. `lds_r` 的 stride 为 `(1, TN+1)`，它和 a1 的 `make_transposed_view` 有什么关系？有什么不同（一个用 stride，一个用变换）？什么情形下只能用变换、不能只改 stride？
3. 用 `make run A=a2 CASE=bandwidth` 记下本作业的比值，与 Lab 1 a2 参考答案（约 0.31–0.35）比较，解释差异（从每个线程一次访问的字节数、访存宽度与合并、LDS 的访问方式等角度；Lab 1 的参考答案里一个线程一次读几个元素？）。

## 动手探索（不判分）

- 把 LDS 的 padding 去掉（两个视图的 stride 里 `TN + 1` 改成 `TN`，`smem` 也改小），带宽变化多少？多跑几次取最好。
- 把 tile 改成 32×32（相应改 encoding），带宽怎样变化？为什么？
- 实现“跳过 LDS”的版本（读入后直接写到跨步的输出视图），与 LDS 版本比较带宽；解释差距来自哪里（全局写是否合并）。

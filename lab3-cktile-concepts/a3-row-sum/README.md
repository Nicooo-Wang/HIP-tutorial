# Lab 3 · a3：用 CK-Tile 做行归约

## 你要做什么

编辑 `a3-row-sum/row_sum.hip`。输入 `in` 是 `rows × cols` 的行主序 fp32 矩阵，对每一行求和：`out[r] = Σ_c in[r*cols + c]`。

```cpp
void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

这是 Lab 3 的收尾作业，**自主设计**：骨架里只有 include 和 `launch_ck_row_sum` 的空函数体，没有 TODO 之外的任何代码。你要自己决定一个 block 处理几行、tile 多宽、行和列怎样映射到 warp / lane / Y，怎样沿列方向滑动窗口，输出怎样写回。用到的 CK-Tile 概念都在讲义 §3.1–3.5 与此前各节中出现过。

约束：

- 一行不能跨 warp（`block_tile_reduce_sync` 只在 wavefront 内合并，见 `ck_tile/ops/reduce/block/block_reduce.hpp` 第 9 行的注释与讲义 §3.2 末尾）。
- 结果方向（行）上至少要留一个 Y 维不被归约（同文件第 12–15 行、第 257–260 行的注释，讲义 §3.1）。
- 行数、列数都可能不是 tile 的整数倍；只有当 `cols` 满足对齐条件时才可以声明向量长度大于 1，而 `cols` 在运行时才知道。
- 不要写数组范围之外的内存，不要修改输入，`stream` 原样传给 kernel 启动语句。

骨架原样提交时所有用例 FAIL（输出保持初始的 NaN），这是预期的。

## 怎么判分

在 `lab3-cktile-concepts/` 目录下运行：

```bash
make grade A=a3                            # 判分
make run A=a3 CASE=ragged_cols             # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `aligned_small` | 2 | 8×256 |
| `many_col_tiles` | 2 | 列数远大于一个 tile 的宽度：窗口必须沿列滑动（16×4096、24×1280；另有 5×260：4 的倍数，但最后一块只剩几列） |
| `ragged_cols` | 3 | cols = 31、257、1001、4099（奇数，cols % 4 为 1 或 3） |
| `cols_mod4_eq2` | 2 | cols = 30、258、4098、1002、6、2（偶数但 cols % 4 == 2）：把“偶数”当作可以承诺向量 4 会在这里出错；若你写了向量 2 的分支，它也只在这里被执行 |
| `ragged_rows` | 2 | 1 行、13 行、1001 行、1×1 |
| `large` | 1 | 4096×4096 |
| `bandwidth_aligned`（性能） | 3 | 8192×8192（cols 是 4 的倍数）：读带宽不低于同进程实测 D2D 拷贝带宽的 80%（只是“不至于慢得离谱”的下限） |

全 1 输入的检查是精确比较（抓漏列、重复列）；随机输入按求和量级给容差，不依赖你的求和顺序。通过性能用例时会打印本次实测拷贝带宽和你的比值。

## Hints

<details><summary>Hint 1（方向）</summary>

- 从“一行只由一个 warp 处理”出发。画出一个 8×256 的 tile：每个 (warp, lane, y0, y1) 对应哪一行、哪一列？
- 一个 lane 一次读几个连续元素？64 个 lane 一次覆盖多少列？列数更大时怎么办？
- 归约分两步：线程内把哪几个 Y 维合并？跨 lane 又合并什么？为什么行方向不能再用 lane 来划分？
- 越界的行、列怎么处理？回忆 a2。输出窗口的分布从哪里来？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- 分布：行方向的 H 段由 warp 和一个 Y 分担，这个 Y 不被归约；列方向的 H 段由 lane 和另一个 Y 分担，这个 Y 是每个 lane 一次读的连续元素，也是被归约的那一个。按讲义 §2.1 的自查清单核对你的 encoding，并确认“一行的所有元素都在同一个 wavefront 里”（§3.2 末尾）。
- 归约分三层：每一块 tile 先在线程内合并（讲义 §3.1 的两个 `block_tile_reduce` 重载：第一块用“返回新累加器”的版本，之后各块用“累加进已有累加器”的版本）；整行走完之后，跨 lane 合并一次（§3.2）；最后把结果写回。想一想跨 lane 合并为什么不该放在循环里。
- 写回：输出是一维的，行数同样可能不整除；输出窗口的分布要和累加器一致，不必自己设计（§3.2）。
- 向量长度：视图的倒数第二个参数必须如实承诺（§3.4）；`cols` 的整除性在主机端判断，按结果启动不同的模板实例（§3.5）。

</details>

<details><summary>Hint 3（卡住很久再看）</summary>

只给步骤顺序，不给代码：

1. 定常量：每 block 几个 wavefront、每个 wavefront 管几行、每个 lane 一次读几个连续元素；由它们算出 tile 的高和宽。向量长度做成 kernel 的模板参数。
2. kernel 里：输入视图（承诺向量长度，两维都 pad 到 tile）→ 分布 → 窗口，原点的行坐标来自 `blockIdx.x`，列坐标从 0 开始。
3. 对第一块建累加器；循环里先挪窗口、再 load、再累加，直到走完 `cols`；循环外做一次跨 lane 合并。
4. 一维的输出视图（同样 pad）+ 借用累加器分布的窗口，store。
5. 主机端：grid 向上取整；按 `cols` 的整除性选模板实例，`stream` 原样传入。

</details>

## 判分覆盖不到的 bug

- **总是用标量访存（向量长度 1）**结果永远正确，也能通过性能用例：本机上标量版约 1.04–1.13 倍拷贝带宽，向量版约 1.7，阈值只有 0.80。判分不检查你是否向量化，请自己用 `make run A=a3 CASE=bandwidth_aligned` 对比两种配置。
- 性能用例只在 8192×8192 一种规模、全 1 输入下测量。
- 一行跨 warp 的设计（需要经 LDS 合并）不被要求，也不会被专门检查。

## 书面问题（写到 ../answers.md）

1. 为什么 `cols % 4 ≠ 0` 时声明向量长度 4 会出错？（提示：一个从 4 的倍数列开始的向量，在 `cols % 4 ≠ 0` 时会不会伸进下一行？pad 判断有效性时看的是向量的哪个元素？在 gfx942 上行首不对齐并不会读错，讲义 §3.4 有实测；对齐只是在别的硬件上可能出现的额外问题。）
2. `block_tile_reduce_sync` 之后，同一行的结果被 R 段上的多个 lane 同时持有（你的设计里是几个？）。`store_tile` 时，这些 lane 会不会都去写同一个地址？读源码回答，不要猜：R 维在分布的坐标适配器里是哪种变换、它对算出的 X 坐标有没有贡献（`core/tensor/tile_distribution.hpp` 第 285–292 行，`replicate` 变换的定义在 `core/algorithm/coordinate_transform.hpp`）；`store_tile`（`core/tensor/store_tile.hpp`）→ `tile_window.hpp` 的 `store()` 这条路径上，有没有任何按 R 坐标筛选写入者的逻辑？再设计一个小实验验证结论（提示：sync 之后、store 之前，给每个 lane 的值加上一个与 lane id 有关的标记，看最后留在内存里的是谁的值）。最后说明：为什么这样写在本作业里是无害的，又为什么 sync 必须放在 store 之前。
3. 如果要支持一行跨 warp（例如 cols 很大而 rows 很小），需要额外做什么？（提示：跨 warp 归约要经过 LDS，类似 Lab 1 a3。）
4. 设计说明：用 3–5 句话写下你的 block 形状、每 block 行数、tile 宽度、Y 维划分、向量长度的选择和理由。

## 动手探索（不判分）

- 把“总是向量 1”和“按 `cols % 4` 分派”的带宽比较一下；再试 `cols = 8190`（不是 4 的倍数）会走哪条路径。
- 每 block 的行数（Y0 重复次数）改成 1、4，带宽怎样变化？

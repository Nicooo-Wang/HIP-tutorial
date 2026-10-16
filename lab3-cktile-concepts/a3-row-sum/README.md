# Lab 3 · a3：用 CK-Tile 做行归约

## 你要做什么

编辑 `a3-row-sum/row_sum.hip`。输入 `in` 是 `rows × cols` 的行主序 fp32 矩阵，对每一行求和：`out[r] = Σ_c in[r*cols + c]`。

```cpp
void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

这是 Lab 3 的收尾作业，**自主设计**：骨架里只有 include 和 `launch_ck_row_sum` 的空函数体，没有 TODO 之外的任何代码。你要自己决定一个 block 处理几行、tile 多宽、行和列怎样映射到 warp / lane / Y，怎样沿列方向滑动窗口，输出怎样写回。用到的 CK-Tile 概念都在讲义 §3.1–3.5 与此前各节中出现过。

约束：

- 一行不能跨 warp（`block_tile_reduce_sync` 的限制，见 `ck_tile/ops/reduce/block/block_reduce.hpp` 开头的注释）。
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
| `many_col_tiles` | 2 | 列数远大于一个 tile 的宽度：窗口必须沿列滑动 |
| `ragged_cols` | 3 | cols = 31、257、1001、4099（不是 4 的倍数） |
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

- `block_tile_reduce<float>(tile, sequence<1>{}, f, 0.0f)` 返回的张量比 `tile` 少了第 1 个 Y 维；循环中用 4 个参数的重载把后续 tile 累加进去；循环结束后做一次 `block_tile_reduce_sync`；输出窗口的 distribution 用 `acc.get_tile_distribution()`。
- 行方向的 H 段由 warp 和 Y0 分担，列方向的 H 段由 lane 和 Y1 分担；Y0 不被归约（`block_tile_reduce` 要求至少一个 Y 维保留）。
- `make_naive_tensor_view` 的倒数第二个参数是“保证的最内维向量长度”，必须如实声明；用 `cols % 4 == 0` 在运行时选择不同的模板实例。

</details>

## 判分覆盖不到的 bug

- **总是用标量访存（向量长度 1）**结果永远正确，也能通过性能用例：本机上标量版约 1.04–1.13 倍拷贝带宽，向量版约 1.7，阈值只有 0.80。判分不检查你是否向量化，请自己用 `make run A=a3 CASE=bandwidth_aligned` 对比两种配置。
- 性能用例只在 8192×8192 一种规模、全 1 输入下测量。
- 一行跨 warp 的设计（需要经 LDS 合并）不被要求，也不会被专门检查。

## 书面问题（写到 ../answers.md）

1. 为什么 `cols % 4 ≠ 0` 时声明向量长度 4 会出错？（提示：每行起始地址的对齐。）
2. `block_tile_reduce_sync` 之后，同一行的 64 个 lane 都持有结果，`store_tile` 时会重复写吗？读 `store_tile` 相关源码，说明 CK-Tile 如何处理复制（R）维 / 被归约维。
3. 如果要支持一行跨 warp（例如 cols 很大而 rows 很小），需要额外做什么？（提示：跨 warp 归约要经过 LDS，类似 Lab 1 a3。）
4. 设计说明：用 3–5 句话写下你的 block 形状、每 block 行数、tile 宽度、Y 维划分、向量长度的选择和理由。

## 动手探索（不判分）

- 把“总是向量 1”和“按 `cols % 4` 分派”的带宽比较一下；再试 `cols = 8190`（不是 4 的倍数）会走哪条路径。
- 每 block 的行数（Y0 重复次数）改成 1、4，带宽怎样变化？

# Lab 3 · a1：layout —— tensor descriptor 与坐标变换

## 你要做什么

CK-Tile 里的 layout 不是一个公式，而是一个 **tensor descriptor**：一串*坐标变换*（pass_through、merge、unmerge、xor……），把“上层逻辑坐标”一层层翻译成“线性偏移”。同一块内存可以挂上不同的视图（转置、展平、分块、swizzle），全程不搬任何数据。这是之后 LDS 布局和 GEMM tile 划分的基础。

编辑 `a1-layout/layout.hip`，补全 5 个返回 descriptor 的函数（`make_row_major` 已给出作示例）。它们都是 `CK_TILE_HOST_DEVICE constexpr`，测试在设备端对每个上层坐标调用 `calculate_offset`，同时检查上层维度数（`get_num_of_top_dimension()`）和每一维的长度（`get_length(number<k>{})`）。

| TODO | 函数 | 上层坐标 → 偏移 | 讲义 |
|------|------|-----------------|------|
| `a1-1` | `make_col_major<M,N>` | (i, j) → j·M + i | §1.2 |
| `a1-2` | `make_transposed_view<M,N>` | 基于 M×N 行主序，上层 (j, i)，j<N，i<M → i·N + j | §1.3 |
| `a1-3` | `make_flattened<M,N>` | 基于 M×N 行主序，上层 1 维坐标 k → k | §1.4 |
| `a1-4` | `make_tiled<M,N,TM,TN>` | 上层 (I, J, ii, jj) → (I·TM+ii)·N + J·TN + jj | §1.5 |
| `a1-5` | `make_xor_swizzled<M,N>` | (i, j) → i·N + (j ^ (i % N)) | §1.6 |

注意：

- 骨架里每个函数体都放了一个**能编译但不对**的占位（例如没做变换的 row-major、stride 全为 0 的 naive descriptor），所以原样提交时只有 `row_major` 通过，其余用例都 FAIL，这是预期的。写你的实现时把占位整行替换掉。
- 上层维度的**个数、顺序与长度**都是答案的一部分：`flattened` 必须是 1 维、长度 M·N；`transposed_view` 的上层长度是 (N, M)；`tiled` 必须是 4 维，顺序为 (I, J, ii, jj)，长度为 (M/TM, N/TN, TM, TN)。偏移全对、长度写错的 descriptor 拿去配 `pad_tensor_view` 时（它用 `get_length` 作为每一维的原长度来判断越界），越界判断就会错。

## 怎么判分

在 `lab3-cktile-concepts/` 目录下运行：

```bash
make grade A=a1                          # 判分
make run A=a1 CASE=tiled                 # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `row_major` | 1 | 示例函数，两种形状 + 各维长度 |
| `col_major` | 1 | 两种形状（4×8、5×3），所有坐标的偏移 + 各维长度 |
| `transposed_view` | 2 | 三种形状（4×8、6×10、5×5），上层形状为 N×M；各维长度应为 (N, M) |
| `xor_swizzled` | 2 | 8×8 和 16×4 两种形状 + 各维长度 |
| `flattened` | 2 | 4×8，32 个坐标的偏移 + 上层维度数应为 1、长度应为 32 |
| `tiled` | 3 | (M, N, TM, TN) = (8, 12, 4, 3) 与 (12, 8, 4, 4)：所有坐标的偏移 + 上层维度数应为 4 + 各维长度 |

所有用例都是把 descriptor 在全部上层坐标上的偏移与 CPU 公式逐个比较，并检查输出数组两侧的哨兵区没有被写；偏移全对之后再检查各维长度，长度不对时会打印“上层第 k 维的长度期望 …，得到 …”。失败信息里的“行/列”对 2D 用例就是你的上层坐标 (第 0 维, 第 1 维)；`tiled` 的输出按 (I, J, ii, jj) 字典序排成一行，最后一项是上层维度数。

## Hints

<details><summary>Hint 1（方向）</summary>

- naive descriptor 只由“每一维的长度”和“每一维的 stride”决定。列主序和行主序的区别落在哪一个量上？
- `transform_tensor_descriptor` 的四个参数分别是什么：底座、变换列表、每个变换吃下层的哪些维度、每个变换产出上层的哪些维度。转置：两个变换各自只吃一个下层维度，区别只在它们产出的上层维度编号。
- flatten 是“多变一”，tile 是“一变多”，xor 是“多对多”，各对应哪种变换？
- 先用纸笔对一个小例子（如 4×8）把“上层坐标 → 下层坐标 → 偏移”这一步手算出来，再决定每个变换怎么摆。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- a1-1：只有 stride 要换，长度不变。想一想每走一步 i、每走一步 j，内存地址分别前进多少。
- a1-2：变换列表里是两个 pass_through，长度分别对应下层第 0、1 维。第三个参数告诉库“它们吃下层的哪一维”，第四个参数告诉库“它们产出上层的第几维”。要让上层第 0 维是 j，这个编号该怎么给？
- a1-3：merge 变换的参数是“被合并的各维长度”，下层吃两个维度，上层产出一个维度。合并时哪一维是高位、哪一维是低位，由参数顺序决定。
- a1-4：两个 unmerge，各吃一个下层维度。unmerge 的长度列表里，先写的是高位还是低位？产出的上层维度编号用两个 sequence 各写 2 个数，想清楚上层 0..3 分别是 I、J、ii、jj 中的哪个，再决定每个 sequence 里写哪两个数。
- a1-5：xor 变换吃下层两个维度、产出上层两个维度。先去源码 `/opt/rocm/include/ck_tile/core/algorithm/coordinate_transform.hpp` 读 `struct xor_t` 的 `calculate_lower_index`，弄清它把上层哪个分量与哪个分量异或，再决定参数里长度的写法。

</details>

<details><summary>Hint 3（卡住很久再看）</summary>

整体写法：每个函数都是“底座（row-major 或 naive）→ 一次 `transform_tensor_descriptor`”。col_major 不需要变换，直接改 naive 的 stride 即可。其余四个都以 `make_row_major<M, N>()` 为底座，区别只在变换种类（pass_through ×2 / merge ×1 / unmerge ×2 / xor ×1）以及后两个 sequence 参数。

某个用例不通过时：先看“首个错误”的位置和期望值，反推你的实现是把哪个上层坐标映射成了什么偏移；也可以在 `make run` 的结果里对照偏移表，看错的规律（整体转置了？按块错位了？只有某几行错？）。上层维度数不对时，检查最后两个 sequence 参数一共产出了几个上层维度。

</details>

## 判分覆盖不到的 bug

判分只比较偏移、维度数与各维长度，不检查你用了哪些变换。下面这些做法也能拿满分，但请自己分辨它们是不是你想要的：

- **绕开变换直接用 naive descriptor。** 例如 `transposed_view` 直接写长度 (N, M)、stride (1, N) 的 naive descriptor，`tiled` 直接写 4 维长度加手算 stride，结果完全相同。这些是合法的等价写法，但本作业想让你练的是“用变换组合出视图”，因为后续 LDS swizzle 与 tile distribution 都是靠变换组合出来的，手算 stride 做不到 merge 或 xor。
- **`flattened` 不基于 `make_row_major`。** 直接写一维的 naive descriptor `make_naive_tensor_descriptor(make_tuple(number<M*N>{}), make_tuple(number<1>{}))`（长度 M·N、stride 1）也满分；但这样你没有练到 merge 变换，也看不出“展平”和底座的关系。
- 测试只覆盖少数几个形状。如果你的写法依赖了形状的特殊性（例如只在 M == N 时成立），换个形状才会暴露。请自己再试几组 M、N（包括不整除的情形会发生什么？）。
- 测试在设备端的单线程里调用，不涉及性能，也没有检查你的 descriptor 在 `constexpr` 下的编译期行为之外的任何事情。

## 书面问题（写到 ../answers.md）

1. `transposed_view` 没有移动任何数据，它和 Lab 1 a2 中的转置有什么本质区别？什么时候“换视图”就够了，什么时候必须真的搬数据？
2. 用 `xor_swizzled<8,8>` 的偏移表说明：按列读取时，8 个元素落在多少个不同的“bank”上（假设 8 个 bank，bank = offset % 8）？与 row-major 比较。
3. `make_tiled` 得到的 4 维视图如何用来表达“block (I,J) 负责的 tile 中第 (ii,jj) 个元素”？这与 Lab 1 a4 的手写下标计算对应关系是什么？

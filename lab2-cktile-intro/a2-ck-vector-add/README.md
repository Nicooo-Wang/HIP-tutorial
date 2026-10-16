# Lab 2 · a2：用 CK-Tile 写 vector add

## 你要做什么

编辑 `a2-ck-vector-add/ck_vector_add.hip`，在 kernel 里完成 `c[i] = a[i] + b[i]`。这与 Lab 1 a1 的 Part A 是同一件事，但这次不手写“全局下标 + 边界检查”，而是用 CK-Tile 的四件套来*描述*它：

```
tensor view（有形状的全局内存视图）→ tile distribution（tile 内每个元素归哪个线程）
  → tile window（视图上的一个窗口）→ load_tile / sweep_tile / store_tile
```

常量（每 block 4 个 wavefront × 64 lane，每线程 4 个 float，一个 block 处理 1024 个元素）、`va` 的构造和 `launch_ck_vector_add` 里的 grid 计算都已给出，`stream` 原样传给启动语句，不要改：

```cpp
void launch_ck_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream);
```

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a2-1` | `ck_vector_add_kernel` | 仿照 `va` 构造 `vb`、`vc` | §3.1 |
| `a2-2` | `ck_vector_add_kernel` | 写出 `tile_distribution_encoding` 的 6 个模板参数 | §3.2 |
| `a2-3` | `ck_vector_add_kernel` | 计算窗口原点，创建 3 个 tile window | §3.3 |
| `a2-4` | `ck_vector_add_kernel` | `load_tile`、`sweep_tile` 逐元素相加、`store_tile` | §3.4 |

注意：

- 骨架里 `a2-2` 到 `a2-4` 都是空的，原样提交时 `c` 保持初始的 NaN，所有用例都会 FAIL，这是预期的。
- 骨架里**没有**给出任何“能编译的 encoding 占位”：有些写错的 encoding 恰好能算出正确结果（比如只覆盖了部分线程、再由别的线程补上），那样的代码看似通过，其实并不是你想要的分布。请先在纸上画出 1024 个元素怎样分给 4×64 个线程，再动手写。
- 你不需要（也不应该）写 `if (i < n)`：越界处理发生在 tensor view 里，问题 2 会让你去源码里找到它。

## 怎么判分

在 `lab2-cktile-intro/` 目录下运行：

```bash
make grade A=a2                          # 判分
make run A=a2 CASE=one_tile              # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `one_tile` | 2 | n = 1024：恰好一个 tile，用上 block 里全部 256 个线程 |
| `many_tiles` | 2 | n = 1024 × 37：多个 block，窗口原点要随 block 移动 |
| `tiny` | 2 | n = 1、3、64：比一个 tile 小得多 |
| `ragged` | 3 | n = 1023、4097、1000003：最后一个 tile 不满，不能越界写 |
| `large` | 1 | n = 2²⁴ + 5：大数组 |
| `bandwidth`（性能） | 2 | n = 2²⁶：带宽不低于同进程实测 D2D 拷贝带宽的 64% |

所有正确性用例逐位比较结果，并检查数组外的哨兵区没有被写、输入 a 和 b 没有被修改。读失败信息的方法同 Lab 1 a1：“得到 nan”表示该元素没有任何线程写过。`one_tile` 里“前 256 个元素对、后面全错”之类的模式，本身就是线索。

## Hints

<details><summary>Hint 1（方向）</summary>

- 先在纸上算：H = (4, 64, 4) 时，线程 (warp = 1, lane = 2) 负责 tile 里的哪几个元素下标？（把 H 的三段看成“行主序的三个坐标”。）
- 再问自己：这样分完，整个 block 的 256 个线程是不是恰好覆盖了 1024 个元素，一个不多一个不少？
- 窗口原点：第 b 个 block 的窗口应该从全局的第几个元素开始？一个窗口有多大？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- a2-1：`va` 的每个参数含义见讲义 §3.1；`vb`、`vc` 只是换了指针。`c` 是要写的数组，视图类型和只读的一样吗？
- a2-2：encoding 的 6 个模板参数依次是 R、H、P→X、P→H 段、Y→X、Y→H 段（讲义 §3.2）。本题没有复制维；H 里要放几段？**P 维要写几个？** 读 CK-Tile 源码 `/opt/rocm/include/ck_tile/core/tensor/tile_distribution.hpp` 第 97–105 行：当 P 维的个数是 1 个和 2 个时，线程分别用什么来决定自己是谁？对照 block 里有 4 个 wavefront 这件事，想想该选哪种。
- a2-3：窗口的“长度”参数是编译期常量，“原点”是运行期值；`blockIdx.x` 的类型是 unsigned，直接参与乘法会遇到什么编译错误？CLAUDE.md 与讲义 §3.3 都有提示。
- a2-4：`load_tile` 返回的对象怎么按元素读写？输出 tile 需要先用与输入相同的分布创建出来（讲义 §3.4）；`sweep_tile` 的回调参数是什么，怎样既读 `ta`、`tb` 又写 `tc`？

</details>

<details><summary>Hint 3（卡住很久再看）</summary>

整体骨架是：三个 view → 一个 distribution → 三个以同一个 distribution 和同一个原点为参数的 window → 两次 load、一个按分布创建的空 tile、一次 sweep、一次 store。encoding 中 P→X、P→H 段的两个参数各有两项，分别对应 warp id 与 lane id，两项都指向 H 的同一个 X 维，指向的段号分别是 H 的第 0 段、第 1 段；Y 指向的是 H 的最后一段。窗口原点 = block 序号 × 每 block 元素数。

</details>

## 判分覆盖不到的 bug

- 判分检查的是结果和性能，不检查你是否理解了分布：如果你把 kVec、kWarps 改成别的合法值并相应改 encoding，同样能通过。这是允许的，也是问题 1 要你实验的内容。
- 如果你的 encoding 让某些线程重复写同一元素（相同的值），结果仍正确，判分发现不了。请用问题 3 的“类型层面描述”视角，自己检查每个元素有且只有一个负责线程。
- 性能用例只在 n = 2²⁶ 一种规模下测；小 n 下的 launch 开销不在判分范围内。

## 书面问题（写到 ../answers.md）

1. 把 `kVec` 改成 1 或 8（同时检查 `kTile` 等常量是否仍然自洽），`bandwidth` 用例的结果怎样变化？为什么？（可用 `make run A=a2 CASE=bandwidth` 看实际带宽。）
2. n = 1023 时最后一个 tile 不满，为什么代码里不需要写 `if (i < n)`？越界的 load 返回什么？越界的 store 又怎样？（读 `/opt/rocm/include/ck_tile/core/tensor/buffer_view.hpp`，找 `address_space_enum::global` 对应的那个 `buffer_view` 的 `get` 和 `set`。）
3. 比较本作业与 Lab 1 a1 的代码：CK-Tile 把哪些“手写下标计算”变成了类型层面的描述？这种描述换成 2D tile 时的好处是什么？

## 动手探索（不判分）

- 把 `kWarps` 改成 1 或 2（同时改 `kBlock`、`kTile`，encoding 里引用常量的地方会自动跟着变），`bandwidth` 有变化吗？
- 故意把 encoding 改成只有 1 个 P 维（H 为 `<kBlock, kVec>`），用 `make run A=a2 CASE=one_tile` 观察哪些元素出错，验证 Hint 2 里关于 P 维个数的结论。试完记得改回来。

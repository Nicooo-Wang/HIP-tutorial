# Lab 2 书面问题

> 书面问题不计分，但它们是读懂 CK-Tile 源码的第一步，Lab 3 起会反复用到。题目原文见各 Assignment 的 README；需要的背景见讲义 §2（C++）与 §3（四件套）。引用源码时请写明文件路径和行号（源码在 `/opt/rocm/include/ck_tile/`）。

## a1 cpp-meta（a1-cpp-meta/README.md）

### 1. `static_for` 的循环变量为什么不能是普通 `int`？（提示：`number<I>` 能用于模板参数。）

（在此作答）

### 2. `number<3>{} + number<4>{}` 在 CK-Tile 中结果的类型是什么？去 `/opt/rocm/include/ck_tile/core/numeric/integral_constant.hpp` 找到答案，并引用相关代码行。

- 结果类型：
- 相关代码（文件:行号）：

```cpp
// 粘贴你找到的代码
```

（在此作答）

### 3. 为什么 CK-Tile 要把 tile 尺寸等参数放进类型里，而不是作为普通函数参数？

（在此作答）

## a2 ck-vector-add（a2-ck-vector-add/README.md）

### 1. 把 `kVec` 改成 1 或 8（同时检查 `kTile` 等常量是否仍然自洽），`bandwidth` 用例的结果怎样变化？为什么？

命令：

```bash
make run A=a2 CASE=bandwidth   # 输出带宽，下一行是“本次实测拷贝带宽 … GB/s，比值 …”
```

带宽有噪声：每种配置跑 3–5 次，表里填带宽最好的那一次（连同同一次的拷贝带宽与比值）。两种配置相差不到 10% 时，可能只是噪声。

| kVec | 带宽（GB/s，最好一次） | 同次实测拷贝带宽（GB/s） | 比值 |
|------|--------------|--------------------------|------|
| 1    |              |                          |      |
| 4    |              |                          |      |
| 8    |              |                          |      |

- 原因分析：

（在此作答）

### 2. n = 1023 时最后一个 tile 不满，为什么代码里不需要写 `if (i < n)`？越界的 load 返回什么？越界的 store 又怎样？

从 `/opt/rocm/include/ck_tile/core/tensor/buffer_view.hpp` 里 `address_space_enum::global` 对应的那个 `buffer_view` 的 `get` 和 `set` 出发，一路追到 `core/arch/amd_buffer_addressing.hpp` 里真正发出 buffer 指令的函数（讲义 §3.1）。

- 为什么不需要 `if (i < n)`：
- 越界 load 返回（文件:行号）：
- 越界 store 怎样（文件:行号）：

（在此作答）

### 3. 比较本作业与 Lab 1 a1 的代码：CK-Tile 把哪些“手写下标计算”变成了类型层面的描述？这种描述换成 2D tile 时的好处是什么？

（在此作答）

### 4. 把 H 写成 (2, 64, 4)、窗口长度改成 512，其余不变（block 仍是 4 个 wavefront、grid 仍按 1024 一个 block 计算），判分全部通过。为什么输出仍然完全正确？为什么它仍然是 bug？

- warp 2、lane 0 的线程按这个 encoding 拿到的 H 坐标与元素下标：
- 为什么结果碰巧全对：
- 为什么仍然是 bug（换一种 grid 计算或换成 2D tile 会怎样）：

（在此作答）

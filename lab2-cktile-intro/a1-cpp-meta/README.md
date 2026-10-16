# Lab 2 · a1：读懂 CK-Tile 所需的 C++

## 你要做什么

编辑 `a1-cpp-meta/cpp_meta.hip`，完成 6 个小函数。它们都用 CK-Tile 自己的类型（`sequence`、`array`、`tuple`、`number`、`static_for`）练习，读完你就能看懂 CK-Tile 源码里最常见的写法：

```cpp
template <index_t... Is> constexpr index_t product(ck_tile::sequence<Is...>);        // E1：积，空 sequence 为 1
template <index_t... Ls> constexpr auto row_major_strides(ck_tile::sequence<Ls...>); // E2：<4,8,2> → {16,2,1}
template <index_t N> __device__ void fill_squares(float* out);                       // E3：out[i] = i*i
template <class T> constexpr index_t vec16();                                        // E4：16 字节装几个 T
template <class... Ts> constexpr double tuple_sum(const ck_tile::tuple<Ts...>& t);   // E5：异构 tuple 求和
constexpr auto add_numbers = [](auto a, auto b) { /* ... */ };                       // E6：number<> 相加
```

| TODO | 位置 | 内容 | 讲义 |
|------|------|------|------|
| `a1-E1` | `product` | 折叠表达式 | §2.1 |
| `a1-E2` | `row_major_strides` | 编译期数组 + constexpr 循环 | §2.2 |
| `a1-E3` | `fill_squares` | `static_for` 展开循环 | §2.3 |
| `a1-E4` | `vec16` | `if constexpr` | §2.4 |
| `a1-E5` | `tuple_sum` | `tuple::at(number<I>{})` | §2.5 |
| `a1-E6` | `add_numbers` | 泛型 lambda + `number` | §2.6 |

每个 TODO 上方都有一个“同类但不同”的小例子，照着它改写即可。

**编译失败时先看是否丢了 `constexpr`。** 测试里有 `constexpr index_t a = product(...)` 这类编译期调用，如果你把函数改成了非 `constexpr`，整个测试会编译失败、得 0 分，并显示编译错误。这是有意设计：CK-Tile 里非 `constexpr` 的函数同样无法用在模板参数里。同理，`row_major_strides` 的返回类型如果不是 `ck_tile::array<index_t, N>`，测试里的 `static_assert` 会让整个测试文件编译失败。

## 怎么判分

在 `lab2-cktile-intro/` 目录下运行：

```bash
make grade A=a1                    # 判分
make run A=a1 CASE=e2_row_major_strides  # 只运行一个用例，看原始输出
```

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `e1_product` | 2 | `<2,3,4>` 为 24；空 sequence 为 1；单元素 |
| `e2_row_major_strides` | 3 | 返回类型是 `array<index_t, N>`；`<4,8,2>`、`<5>`、`<3,1,7,2>`（含长度为 1 的维度） |
| `e3_static_for_device` | 2 | 设备端 `fill_squares<8>` 与 `<1>`，并检查没有越界写 |
| `e4_if_constexpr` | 2 | float=4、half=8、int8=16、double=2 |
| `e5_tuple_sum` | 2 | 异构 tuple `(1, 2.5f, 3.0)` 的和；单元素 tuple |
| `e6_compile_time_lambda` | 2 | 返回类型必须是 `number<7>`、`number<-5>`（值在类型里） |

## Hints

<details><summary>Hint 1（方向）</summary>

- E1：折叠表达式 `(init op ... op pack)` 里，`init` 要是 `op` 的单位元，空包时才对。
- E2：行主序下“最后一维变化最快”，所以 stride 应该从哪一端开始算？每往外一维，要乘上哪个长度？
- E3：`static_for` 的循环变量是一个类型（`number<I>`），不是整数变量；你需要的编译期常量从哪里取？
- E4：`if constexpr` 只在编译期选分支；`sizeof` 的结果单位是什么？
- E5：tuple 里每个元素类型不同，下标必须是编译期的值；你手上有谁能提供？
- E6：结果的“值”要放进类型里，应该返回哪个模板的哪个实例？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- E1：折叠的初值取“乘法的单位元”，然后把整个参数包向右折叠进去。
- E2：先把参数包放进一个编译期的普通数组，再准备一个累乘变量；从最后一维开始往第 0 维走，每走到一维，先把累乘变量当前值记为这一维的 stride，再把这一维的长度乘进累乘变量。循环下标要能走到第 0 维之后停下，想想该用有符号还是无符号类型。
- E3：在 lambda 里先从循环变量取出编译期常量放进一个 `constexpr` 变量，再用它计算平方并写入 `out` 对应位置。
- E4：double 走一个分支，其余类型走另一个分支；后者用“16 字节”除以“单个元素的字节数”。
- E5：用 `static_for` 遍历 tuple 的全部下标（下标个数由参数包的大小决定），每次取出该下标的元素，转成 double 加到一个局部累加变量上，最后返回它。
- E6：从 `a`、`b` 的类型里分别取出各自编码的值，相加后用这个和构造新的 `number` 类型的对象并返回。

</details>

## 书面问题（写到 ../answers.md）

1. `static_for` 的循环变量为什么不能是普通 `int`？（提示：`number<I>` 能用于模板参数。）
2. `number<3>{} + number<4>{}` 在 CK-Tile 中结果的类型是什么？去 `/opt/rocm/include/ck_tile/core/numeric/integral_constant.hpp` 找到答案，并引用相关代码行。
3. 为什么 CK-Tile 要把 tile 尺寸等参数放进类型里，而不是作为普通函数参数？

## 判分覆盖不到的

- 测试检查的是结果，不检查你有没有用折叠表达式、`static_for` 或 `if constexpr`；写成手工展开也能得分。但这几种写法是后续 Lab 读 CK-Tile 代码时的“日常”，建议按提示的方式写一遍。
- E3 的循环会被完全展开；你可以用 `--save-temps` 看看生成的汇编里有没有循环（动手探索，不判分）。

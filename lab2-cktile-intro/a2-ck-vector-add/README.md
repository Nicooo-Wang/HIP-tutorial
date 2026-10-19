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
- 你不需要（也不应该）写 `if (i < n)`：越界访问由 buffer 指令的硬件 range 检查屏蔽（range 由视图长度 n 决定，讲义 §3.1），问题 2 会让你去源码里找到它。

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
| `bandwidth`（性能） | 2 | n = 2²⁶：比值（= 带宽 / 同进程实测 D2D 拷贝带宽，Lab 1 §1.3）不低于 0.55（55%）。注意这个 D2D 拷贝（hipMemcpy device-to-device）只是自校准的基准，不是硬上限：写得好的 kernel 可以超过它（参考答案的比值约 1.45）。`make run A=a2 CASE=bandwidth` 会同时打印本次带宽、同次实测拷贝带宽和比值 |

所有正确性用例逐位比较结果，并检查数组外的哨兵区没有被写、输入 a 和 b 没有被修改。读失败信息的方法同 Lab 1 a1：“得到 nan”表示该元素没有任何线程写过。`one_tile` 里“前 256 个元素对、后面全错”之类的模式，本身就是线索。

## Hints

<details><summary>Hint 1（方向）</summary>

- 先在纸上定 H：tile 是一维的 1024 个元素，要拆成几段？“第几个 wavefront”“第几个 lane”“线程内第几个”各是一段的话，每段的长度分别是骨架里的哪个常量、按什么顺序排（讲义 §3.2 清单第 4、5 条）？
- 拆好后算一算：线程 (warp = 1, lane = 2) 负责 tile 里的哪几个元素下标？（把 H 的各段看成“行主序的坐标”。）
- 再问自己：这样分完，整个 block 的 256 个线程是不是恰好覆盖了 1024 个元素，一个不多一个不少？
- 窗口原点：第 b 个 block 的窗口应该从全局的第几个元素开始？一个窗口有多大？

</details>

<details><summary>Hint 2（关键步骤）</summary>

- a2-1：`va` 的每个参数含义见讲义 §3.1；`vb`、`vc` 只是换了指针。`c` 是要写的数组，视图类型和只读的一样吗？
- a2-2：encoding 的 6 个模板参数依次是 R、H、P→X、P→H 段、Y→X、Y→H 段（讲义 §3.2）。本题没有复制维；H 只有一个 X 维，按 (kWarps, kLanes, kVec) 拆成三段，从高位到低位依次是“第几个 wavefront、第几个 lane、线程内第几个”。**P 维要写几个？** 读 CK-Tile 源码 `/opt/rocm/include/ck_tile/core/tensor/tile_distribution.hpp` 第 97–105 行：当 P 维的个数是 1 个和 2 个时，线程分别用什么来决定自己是谁？对照 block 里有 4 个 wavefront 这件事，想想该选哪种。
- a2-3：窗口的“长度”参数是编译期常量，“原点”是运行期值；`blockIdx.x` 的类型是 unsigned，直接参与乘法会遇到什么编译错误？讲义 §3.3 有提示。
- a2-4：`load_tile` 返回的对象怎么按元素读写？输出 tile 需要先用与输入相同的分布创建出来（讲义 §3.4）；`sweep_tile` 的回调参数是什么，怎样既读 `ta`、`tb` 又写 `tc`？

</details>

<details><summary>Hint 3（卡住很久再看）</summary>

整体骨架是：三个 view → 一个 distribution → 三个以同一个 distribution 和同一个原点为参数的 window → 两次 load、一个按分布创建的输出 tile、一次 sweep、一次 store。窗口原点 = block 序号 × 每 block 元素数。

encoding 还是写不出来时，不要猜语法，回到讲义 §3.2：把“R 维”例子的 6 个参数逐个读懂（它也有 2 个 P 维：每个 P 维在 P→major、P→minor 里各占一个 sequence，顺序固定是先 warp 后 lane），然后用清单逐条核对你自己的 6 个参数——H 的每一段由谁认领、各认领一次。写完用 `make run A=a2 CASE=one_tile` 看出错元素的位置，对照讲义 §3.2 的坑判断是哪一段没认领对。

</details>

## 判分覆盖不到的 bug

判分只看结果是否正确、带宽是否达标，不检查你的分布本身。下面这些写法都能拿满分，但都不是你想要的，请自己检查：

- **P 维认领的段长与 block 对不上，但碰巧全覆盖。** 例如 block 明明有 4 个 wavefront，H 却写成 (2, 64, 4)、窗口长 512，结果一个元素都不缺；反过来把 warp 那一段写得比 wavefront 数大（如 (8, 64, 4)、窗口 2048），也能全对。这两种都是 bug：每个线程负责哪些元素已经不是你设计的那样，换一种 grid 计算或改成 2D tile 就会出错。书面问题 4 会让你分析第一种为什么“碰巧”是对的。
- **视图承诺的向量长度与分布的 Y 段长度不一致。** 视图写 `number<1>{}`、`number<8>{}` 之类，或者干脆不写最后两个参数，结果照样正确。区别只在访存宽度（讲义 §3.1、§3.2）；不写或写 1 时退化成一次 1 个元素的访问，带宽仍可能高于 55% 的门槛。
- **向量那一段不在 H 的最低位**（例如 H 写成 (kVec, kWarps, kLanes) 并让 Y 认领第 0 段）。这是功能正确的另一种分布，但每个线程的元素不再连续，判分不检查这一点。
- 如果你的 encoding 让某些线程重复写同一元素（相同的值），结果仍正确，判分发现不了。请用问题 3 的“类型层面描述”视角，自己检查每个元素有且只有一个负责线程。
- 把 kVec、kWarps 改成别的合法值并相应改 encoding，同样能通过。这是允许的，也是问题 1 要你实验的内容。
- 性能用例只在 n = 2²⁶ 一种规模下测；小 n 下的 launch 开销不在判分范围内。

## 书面问题（写到 ../answers.md）

1. 把 `kVec` 改成 1 或 8（同时检查 `kTile` 等常量是否仍然自洽），`bandwidth` 用例的结果怎样变化？为什么？（`make run A=a2 CASE=bandwidth` 会打印本次带宽、同次实测拷贝带宽和比值。带宽有噪声：每种配置跑 3–5 次、记最好的一次；两种配置相差不到 10% 时，可能只是噪声，如实记录即可。）
2. n = 1023 时最后一个 tile 不满，为什么代码里不需要写 `if (i < n)`？越界的 load 返回什么？越界的 store 又怎样？（从 `/opt/rocm/include/ck_tile/core/tensor/buffer_view.hpp` 里 `address_space_enum::global` 对应的那个 `buffer_view` 的 `get` 和 `set` 出发，一路追到 `core/arch/amd_buffer_addressing.hpp` 里真正发出 buffer 指令的函数：range 是怎么设置的？对 naive 视图，`get`/`set` 收到的“是否有效”标志是什么值？讲义 §3.1。）
3. 比较本作业与 Lab 1 a1 的代码：CK-Tile 把哪些“手写下标计算”变成了类型层面的描述？这种描述换成 2D tile 时的好处是什么？
4. 把 H 写成 (2, 64, 4)、窗口长度改成 512，其余不变（block 仍是 4 个 wavefront、grid 仍按 1024 一个 block 计算），判分全部通过。为什么输出仍然完全正确？为什么它仍然是 bug？（提示：算出 warp 2、3 的线程按这个 encoding 拿到的 H 坐标和元素下标；再想想换一种 grid 计算或换成 2D tile 时会怎样。）

## 动手探索（不判分）

- 把 `kWarps` 改成 1 或 2（`kBlock`、`kTile` 由它推出，会自动跟着变；encoding 里引用常量的地方也一样），`bandwidth` 有变化吗？同样多跑几次再比较。
- 故意把 encoding 改成只有 1 个 P 维（H 为 `<kBlock, kVec>`），用 `make run A=a2 CASE=one_tile` 观察哪些元素出错，验证 Hint 2 里关于 P 维个数的结论。试完记得改回来。

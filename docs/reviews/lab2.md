# Lab 2（lab2-cktile-intro）审查记录

三份独立审查（提示词见 [PROMPTS.md](PROMPTS.md)，`{LAB}` = `lab2-cktile-intro`），审查基线为提交 a891611。
修复一批完成（Task 13 fix）：

| 提交 | 内容 |
|---|---|
| 7bed3a9 | a1：e4 补 bf16/int32/int64，e5 补 double 精度用例，e6 补 (5,6)、(-2,-3)、(7,0)；新增 mutant m4–m6；constexpr 说明与 E4 注释修正 |
| 4e73cf7 | a2：bandwidth 用例通过时也在 msg 里带上拷贝基准与比值（`make run` 可见）；Hint 1/3 收紧；书面问题 1 加“多跑取最好”；问题 2 指向 amd_buffer_addressing.hpp；新增“判分覆盖不到”的三类盲区与书面问题 4；新增 mutant m4、m5 |
| b82bb6b | 讲义按 R3 修订（越界屏蔽机制、guarantee 参数、dword 粒度、向量长度规则、P 维错误不是编译错误等）；§3.1–§3.4 各补一行对应 TODO；CLAUDE.md 补 4 条 CK-Tile 已知坑 |

## R1 学生视角走查

结论：a1 13/13、a2 12/12，均一次编译通过，无阻塞。

| 发现 | 级别 | 处理 |
|---|---|---|
| answers.md a2 问题 1 表格要填“同次实测拷贝带宽”和“比值”，但 `make run` 只打印带宽 | 重要 | 已修（4e73cf7）：test.cpp 在通过时把“本次实测拷贝带宽 … GB/s，比值 …”写进 msg；harness 的 in-process 模式本来就会打印非空 msg，`make grade` 只在失败时打印 msg，所以判分输出不变、`--case` 协议不变、harness 未改动。README 用例表与问题 1、answers.md 都写明了这一点 |
| 泄题：§3.4 完整 scale kernel 与 a2-1/3/4 逐行对应；README Hint 1 直接给 H = (4, 64, 4)；Hint 3 几乎拼出 encoding | 重要 | 部分修改：保留 §3.4 完整例子（第一个 CK-Tile kernel，需要看到四件套如何拼在一起；它是 1 个 wavefront、1 个 P 维、kRepeat = 2，与 a2 的核心 a2-2 不同；a2-1/3/4 本来就是按例子类比的难度 1）。收紧的是：Hint 1 改为问“H 拆几段、每段长度是哪个常量”，H 的三段移到 Hint 2；Hint 3 去掉 P→major/minor 的具体写法，改为“读懂 §3.2 R 维例子的 2 个 P 维写法 + 按清单核对 + 用 one_tile 输出定位”；§3.4 末尾 warn 框中“两个输入各 load 一次、同一个 idx 读两个写一个”的配方改成引导问题（4e73cf7、b82bb6b） |
| kVec 对比实验噪声大（kVec=8 可能比 4 还慢），学生会得出无法解释的结论 | 重要 | 已修（4e73cf7）：README 问题 1、answers.md 表格都改为“每种配置跑 3–5 次取最好；相差 10% 以内可能只是噪声，如实记录”；动手探索的 kWarps 实验同样提示 |
| §3.1–§3.3 末尾没有 assignment 卡片 | 重要 | 核对 Lab 1 约定：卡片一个 Assignment 一张、放在该 Assignment 可做的那一节末尾，Lab 2 的两张卡片（§2 末、§3.5 末）符合约定。但 Lab 2 §2.1–§2.6 每节末尾都有一行 `sec-assign`，§3.1–§3.4 没有，前后不一致，已补“对应 a2 的 a2-1 / a2-2 / a2-3 / a2-4”四行（b82bb6b） |
| 讲义没显式总结“P→major tuple 第 i 项对应第 i 个 P 维、顺序固定 (warp, lane)” | 轻微 | 已在 Hint 3 中写明（4e73cf7）；讲义 §3.2 的 6 参数列表与“P 维有几个”段已分别给出两部分 |
| 小部件 3.1 不输出对应 encoding | 小 | 未改：小部件显示任意 (W, L, V) 的映射已足够验证纸面推导；再输出 encoding 代码就等于给出 a2-2 答案 |
| a1 Hint 2 对 E2、E5 接近伪代码 | 小 | 未改：d0d3ed6 已把 Hint 2 改成不含代码的步骤描述，属于“卡住时看”的层级 |
| a2 骨架 `(void)va;` 不删也能过 | 小 | 未改：无影响 |
| E4 `16 / sizeof(T)` 是 size_t 隐式转 index_t，讲义没提 | 小 | 未改：可编译，值域无风险 |

## R2 判分对抗审查

结论：71 个变体中 0 个正确实现被误判；漏网错误 a1 3 类、a2 4 类，均为中低严重度。

| 发现 | 处理 |
|---|---|
| a1 e5 float 累加器 / 元素先转 float（w_sum_float、w_float_elem）满分 | 已修（7bed3a9）：新增 `(1e-9, 1.0, 1e-9)` 与 `(0.1, 0.2)`，与 double 结果比较（容差 1e-12，避免对求和顺序不同的正确写法误判）。复测：w_sum_float、w_float_elem、w_sum_int 均 11/13；新增 mutant m4_tuple_sum_float |
| a1 e4 只测 4 种类型（w_vec_bf） | 已修（7bed3a9）：补 `bf16_t`=8、`int32_t`=4、`int64_t`=2。w_vec_bf 11/13；新增 mutant m5_vec16_whitelist |
| a1 e6 用例太少，`|`、`|a|+b` 恰好成立（w_add_or、w_add_abs） | 已修（7bed3a9）：补 (5,6)→11、(-2,-3)→-5、(7,0)→7。两者均 11/13；新增 mutant m6_add_bitor |
| a1 e3 用普通 for 代替 static_for（w_plainfor）满分 | 未改：黑盒测试无法区分，README“判分覆盖不到的”已说明 |
| a1 e2 空 sequence 未测 | 未改：空 sequence 的 stride 无意义，审查者也建议不处理 |
| a2 #1 P0 段长小于 / 大于 wavefront 数但仍全覆盖（w_partial H=(2,64,4)、w_oversize H=(8,64,4)）满分 | 搁置为已知盲区（黑盒无法判断分布与线程数是否一致；这样的 mutant 会通过判分，不能放进 verify）。README“判分覆盖不到的 bug”用白话写出；answers.md / README 新增书面问题 4：为什么 (2,64,4) + 512 窗口仍全对、为什么仍是 bug（4e73cf7） |
| a2 #2 视图声明的向量长度与 Y 长度不一致（w_vv1/8/16/novec）满分 | 搁置为已知盲区：要抓 vv1 需把阈值提到约 1.0，会误伤合法的 kVec=1/2。README 盲区一节写明；讲义 §3.1/§3.2 现在讲清了不写或写错时向量宽度怎么变 |
| a2 #3 Y 不在最内层（w_dist_all_ykmajor、w_noncontig_y）满分 | 不是 bug（功能正确的替代分布）；README 盲区一节说明判分不检查“每线程连续 kVec 个元素” |
| a2 `tc(idx) +=` 累加到未初始化寄存器（w_tc_pluseq_uninit） | 搁置：未定义行为恰好为 0，不可确定性观测；讲义 §3.4 现在明确写出 `make_static_distributed_tensor` 不初始化 |
| 建议新增 a2 mutant | 已加（4e73cf7）：m4_view_padded_to_tile（c 视图长度补齐到整 tile，被 tiny、ragged、large 抓住）、m5_wb_origin_zero（被 many_tiles、ragged、large 抓住）。w_partial 不能作为 mutant（会通过） |

复测正确写法（修改后的 test.cpp）：a1 ok1、ok2、ref 13/13；a2 ok_elementwise、ok_inplace_a、ok_kv1/2/8、ok_kw1/2/8、ok_multiY、ok_ref、ok_threadbuf、ok_two_tiles 全部 12/12（R2 目录中的 ok_single_p 实为单 P 维错误写法，2/12，与 R2 报告“单 P 维被抓”一致）。

## R3 技术准确性审查

结论：错误 3、不严谨 7、建议 7。全部在 b82bb6b（讲义、CLAUDE.md）与 7bed3a9 / 4e73cf7（README、骨架注释、answers.md）中处理，改动的结论都在本机重新实测。

| 发现 | 处理 |
|---|---|
| E1 §3.1 称 naive 视图在软件层也做越界判断 | 已修：越界屏蔽完全由硬件 range 完成；`coordinate_has_valid_offset_assuming_top_index_is_valid` 假定顶层坐标合法，只对 pad 等变换起作用，naive 视图下为编译期 `true`。复测：越界 load/store 的 ISA 无 `v_cndmask`，只有 `buffer_*_dwordx4 … offen` |
| E2 §3.1 称省略最后两个 guarantee 参数时 CK-Tile 自己推导、指令相同 | 已修：不推导，`ScalarPerVector` = 1、访问次数 ×4；1D 例子汇编相同是后端合并标量 buffer 访存，不可依赖。复测 ScalarPerVector：(4,1)→4、省略→1、只给 4→1、stride 写 number<1>+只给 4→1、(8,1)→4、(2,1)→2；LLVM IR 中省略版本为逐个 `raw.buffer.load.f32`，带参数版本为 `v4f32` |
| E3 §3.5 称“P 维个数不对”会在编译期拦下 | 已修：只拦 NDimP ∉ {1, 2}；1 与 2 选错能编译、静默出错。同时采纳建议 3（运行期部分还有视图指针与长度） |
| N1 向量维选择规则过于简化 | 已修：安全向量长度 = gcd(视图承诺, Y 段长)，要求承诺 stride 为 1；给出承诺 8/2/不承诺时的实测结果 |
| N2 图 3.2 “按每个元素检查” | 已修：图中文字改为按 dword 检查；新增 pitfall 框，fp16 n = 505 时合法元素 504 读回 0、写被丢弃，n = 506 正常（复测一致） |
| N3 问题 2 与 §3.1 指错阅读位置 | 已修：讲义 §3.1、README 注意事项与问题 2、answers.md 都改为从 buffer_view 的 get/set 追到 amd_buffer_addressing.hpp，并说明越界由硬件 range 完成 |
| N4 §2.2 报错原文与 E2 场景不符 | 已修：改为函数模板实际的 `constexpr variable 's' must be initialized by a constant expression` + `cannot refer to element 18446744073709551615 …`，并说明下标先转 index_t 时为 -1（复测 e2b/e2） |
| N5 §2.3 `k.value` “无论什么都是常量表达式” | 已修：限定为按值传入；引用参数指向 §2.6 |
| N6 “丢 constexpr 整个测试会编译失败”对 E6 不成立 | 已修：骨架第 2–3 行与 a1 README 改为 E1、E2、E4、E5，并说明 E6 lambda 隐式 constexpr |
| N7 §4.2 “两层 static_for” | 已修：三层（247、252、263 行），并说明 `load_tile` 先经第 113–121 行的无参 `load()` |
| 建议 1 §1.2 漏了 `utility/` | 已修 |
| 建议 2 §4.1 行号 471 → 461 | 已修 |
| 建议 3 §3.5 运行期部分 | 已修（并入 E3） |
| 建议 4 `make_static_distributed_tensor` 未初始化 | 已修：§3.4 说明、§3.4 代码注释、§3.5 表格、README Hint 3 去掉“空 tile”说法 |
| 建议 5 README 动手探索“同时改 kBlock、kTile” | 已修：说明二者由 kWarps 自动推出 |
| 建议 6 E4 的 double 分支多余，误导 | 部分处理：保留题目与判分（改规则要动测试、参考答案和 mutant），但骨架注释改为“double 分支只是为了练习 if constexpr 的写法——想一想去掉它结果会不会变”，不再声称 double 是特例 |
| 建议 7 static_for 一般情形走 `static_for_impl` | 已修：§2.1、§2.3 注明逗号折叠只属于 `<0, N, 1>` 特化，一般情形用第 34 行的 `swallow{…}` 包展开 |

## 判分变化在文档里的同步

- a1 README 用例表（e4、e5、e6 新增内容）与 constexpr 说明。
- a2 README 用例表 bandwidth 行、书面问题 1/2/4、动手探索、判分盲区；answers.md 问题 1 表格说明、问题 2 阅读路径、新增问题 4。
- 讲义 §3.4 末尾 warn 框与 §3.5 表格去掉“空 tile”说法。

## 已知并接受的取舍

- §3.4 保留完整 scale kernel（理由见 R1 第 2 条）；§3.5 对照表（规格指定内容）仍说明了 a2-4 的步骤。a2 的核心推导 a2-2（2 个 P 维、warp 段）在讲义和 README 中都没有给出代码。
- a2 判分不检查分布本身（P 段长、向量长度、Y 位置），只在 README 与书面问题里要求学生自查。
- 带宽用例阈值未改（0.55）。

## 校准

### a2 ck-vector-add（kAddRatio = 0.55）

所有比值 = 带宽 / 同进程实测 D2D 拷贝带宽；空闲 MI308X（GPU 1、5 各 4 次，Task 12）。

| 实现 | 带宽 | 比值 |
|---|---|---|
| 拷贝带宽 | GPU 1 约 1970–1996 GB/s，GPU 5 约 1884–1904 GB/s | — |
| 参考答案（kWarps=4、kVec=4） | 2881–3116 GB/s | S_min = 1.45（1.45–1.65） |
| 慢 mutant m3（grid = 1 循环） | 25.7–25.8 GB/s | N_max = 0.013 |
| kVec = 1 | 1654–1670 GB/s | 0.84（GPU 1）/ 0.885（GPU 5） |
| kVec = 2 | 2273–2349 GB/s | ≥ 1.17 |
| kVec = 8 | 2790–2864 GB/s | ≥ 1.45 |
| kWarps = 1 / 2 | 3585–3763 GB/s | > 1.8 |

取值经过：按规则 floor2(S/1.67) = 0.86 会挂掉合法的 kVec=1（0.84）→ 0.64 → 控制者裁定 0.55，使 kVec=1 留约 1.53× 余量、参考答案约 2.6×、慢版本约 42×。比值大于 1 是因为 3 × 256 MiB 的读写模式与 D2D 拷贝不同，拷贝带宽只是归一化的尺子，不是上限。本批复测（GPU 6，修改后的 test.cpp）：参考答案 3376–3502 GB/s，拷贝 1927–1944 GB/s，比值 1.75–1.80。

a1 无性能用例。

## 本批复测的其他实测

- guarantee 参数与 `ScalarPerVector`、IR 中的标量 / 向量 buffer 调用、越界 ISA 无 `v_cndmask`、fp16 dword 粒度、E2 报错原文：见上方 R3 表。
- §3.4 完整 kernel（只改了注释）从讲义自动抽取后重新编译运行：n = 1、3、64、511、512、1023、4097、1000003 全部逐位正确、两侧哨兵完好。“节选自 §3.4”的片段仍是完整 kernel 的子串；12 处“读源码”摘录与 ROCm 7.2 源码逐行一致。
- 浏览器（Playwright，浅色 / 深色 / 390 px）：console 错误 0、目录 21 项、2 张卡片、小部件交互正常、无横向溢出；图 3.2 新文字宽 566 / 640。

## 最终 `make verify`（4e73cf7 起的判分代码）

```
== a1-cpp-meta
   solution    满分 13/13
   mutant      m1_product_zero_init.hip         被抓住: e1_product
   mutant      m2_strides_colmajor.hip          被抓住: e2_row_major_strides
   mutant      m3_vec_width_bits.hip            被抓住: e4_if_constexpr
   mutant      m4_tuple_sum_float.hip           被抓住: e5_tuple_sum
   mutant      m5_vec16_whitelist.hip           被抓住: e4_if_constexpr
   mutant      m6_add_bitor.hip                 被抓住: e6_compile_time_lambda
   skeleton    0/13
== a2-ck-vector-add
   solution    满分 12/12 bandwidth=3536.1GB/s
   mutant      m1_single_p_dim.hip              被抓住: one_tile, many_tiles, ragged, large, bandwidth
   mutant      m2_origin_missing_tile.hip       被抓住: many_tiles, ragged, large, bandwidth
   mutant      m3_single_block_loop.hip         被抓住: bandwidth
   mutant      m4_view_padded_to_tile.hip       被抓住: tiny, ragged, large
   mutant      m5_wb_origin_zero.hip            被抓住: many_tiles, ragged, large
   skeleton    0/12

VERIFY OK
```

harness 未改动，Lab 0 / Lab 1 无需重新 verify。

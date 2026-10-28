# Lab 3（lab3-cktile-concepts）审查记录

三份独立审查（提示词见 [PROMPTS.md](PROMPTS.md)，`{LAB}` = `lab3-cktile-concepts`），审查基线为提交 f42f949。
修复一批完成（Task 17 fix）：

| 提交 | 内容 |
|---|---|
| a224922 | a1：每个用例在偏移之后再检查各维长度（`get_length`）；新增 transposed 5×5、tiled (12, 8, 4, 4) 两个方阵/方块形状；新增 mutant m5_transpose_lengths_wrong；README 源码路径改为 `core/algorithm/`；TODO 点名四个变换构造函数 |
| e0ac0c7 | a3：新增用例 `cols_mod4_eq2`（cols = 30、258、4098、1002、6、2），`many_col_tiles` 加 5×260；新增 mutant m4_dispatch_mod2_vec4；Hint 2 改为纯文字、新增折叠的 Hint 3（步骤顺序）；书面问题 1 的提示改为“向量跨行”；问题 2 去掉“CK-Tile 如何处理 R 维”的预设，改为指向源码与实验；性能失败信息不再暗示判分检查向量化；answers.md 同步 |
| 62d59eb | a2：README“判分覆盖不到”补上向量长度承诺过度（输入 vec4 + 奇数列、LDS 非对齐向量写）这类“碰巧正确”的未定义行为 |
| f0d0fa6 | 讲义：§1.3/§1.4/§1.6 例子换成与 a1-2/3/5 结构不同的问题；§3.2 的完整 row-max kernel 改成三个孤立的 API 片段；R3 的错误与不严谨全部修正 |
| 384e214 | CLAUDE.md“CK-Tile 已知坑”：range 大小、只有内层维越界会绕行、归约两条限制、向量承诺只查首元素，合并重复的 1-P 条目；Lab 2 讲义 §3.1 “比较指令”改为“选择指令 `v_cndmask`” |

## R1 学生视角走查

结论：a1 11/11、a2 12/12、a3 13/13，均一次编译通过、无阻塞；一次通过本身说明泄题偏重。

| 发现 | 级别 | 处理 |
|---|---|---|
| I-1 a3“自主设计”名不副实：§3.2 完整 row-max kernel + §3.3 scale_strip + §3.5 分派 + Hint 2 拼起来就是答案 | 重要 | 按裁定 R21 修改（f0d0fa6、e0ac0c7）：§3.2 的完整 kernel 换成三个孤立片段（线程内归约、跨 lane 合并、写回），每个片段写明输入、输出与分布的变化，不再给出视图/窗口/循环/分派的整体拼装；Hint 2 只用文字描述（无函数签名）；新增折叠的 Hint 3，只给步骤顺序。四条约束（至少留一个 Y、不跨 wavefront、如实承诺向量长度、运行时分派）仍在讲义 §3.1、§3.2 末尾、§3.4、§3.5 讲清。§3.3 scale_strip 与 §3.5 分派函数保留（它们是逐元素例子，讲的是窗口移动与模板分派本身） |
| I-2 a1 的 a1-2/3/5 与 §1.3/§1.4/§1.6 例子近乎同构 | 重要 | 已修（f0d0fa6）：§1.3 改为“按新上层顺序列变换、置换写在第 3 个参数”（图 1.3 重新接线），并以练习形式提到另一种写法，不给答案；§1.4 改为合并*不相邻*的 (a, c) 两维，偏移表重算；§1.6 的 xor 输出挂到新上层 (g, i, v)，讲清“第 3 个参数决定谁打乱谁、第 4 个参数只决定位置”。§1.5 保留（R1 认为 a1-4 尺度合适） |
| I-3 a3 书面问题 1 的提示“行首对齐”与 §3.4 实测矛盾 | 重要 | 已修（e0ac0c7、f0d0fa6）：README 与 answers.md 的提示改为“向量会不会伸进下一行、pad 只看向量第一个元素；gfx942 上不对齐照样读对”；answers.md 补一问“为什么最后一行反而可能是对的”；§3.4 把“列数是 V 的倍数”与“首地址对齐”拆成两件事 |
| I-4 a3 书面问题 2 讲义无法回答，题干预设可疑 | 重要 | 已修：核对源码，`store_tile` → `tile_window.hpp` 的 `store()` 路径没有任何 R 相关逻辑；R 维在分布适配器里是 `replicate` 变换（`tile_distribution.hpp` 第 285–292 行），下层维度数为 0，对 X 坐标没有贡献，所以 R 段上的每个 lane 都写同一个地址（R3 用标记实验证实）。题干改为“会不会重复写”，指向这两处源码并要求设计标记实验；讲义 §3.2 片段 ③ 给出 `replicate` 的线索但不给结论 |
| S-1 a1 README 源码路径 `core/tensor/coordinate_transform.hpp` 不存在 | 小 | 已修：`core/algorithm/coordinate_transform.hpp`；骨架 a1-5 TODO 同步写全路径 |
| S-2 answers.md a3 Q1 未写目录 | 小 | 已修：写全路径 |
| S-3 “视图 C”只在 SVG 里 | 小 | 已修：§2.4 正文用一段话点名视图 C，stride 仍留作思考 |
| 知识缺口：一维输出视图能否 pad | — | 已补：§3.2 片段 ③ 说明一维视图同样可以用 `pad_tensor_view`（各一项） |
| 知识缺口：累加重载要先有 acc | — | 已补：§3.1 讲清两个重载（都是 4 个参数），以及传入的 acc 分布与初值的要求 |
| S-4、S-5 | 记录 | 无需改动 |

## R2 判分对抗审查

结论：a1 26、a2 45、a3 36 个变体；0 个正确实现被误判；漏网 2 类（a3 的 cols%4==2 / 向量 2 分支，a1 的各维长度）。

| 发现 | 处理 |
|---|---|
| a3 `cols % 4 == 2` 从未测到：`cols % 2 == 0 ? vec4` 与三路分派中 vec2 分支的 bug 满分 | 已修（e0ac0c7）：新增 `cols_mod4_eq2`（2 分），总分 13 → 15；新增 mutant m4_dispatch_mod2_vec4。复测 R2 变体：m_dispatch_mod2_vec4、m_vec2_path_bug、m_vec2_path_bug_unrolled 均 13/15（只挂新用例），m_dispatch_ge4_vec4 10/15 |
| a1 不检查各维长度：col (N, M)、transposed (M, N)、flattened M+N、tiled 长度错位均满分 | 已修（a224922）：每个用例偏移全对后检查维度数与各维长度（失败信息“上层第 k 维的长度期望 …，得到 …”）；新增 mutant m5_transpose_lengths_wrong（pass_through 长度写反，偏移不变）。复测：c_col_lengths_swapped 10/11、l_trans_lengths_wrong 9/11、l_flat_length_wrong 9/11、l_tiled_naive_lengths_wrong 8/11 |
| a2 输入视图 vec4 + 奇数列、LDS 行间距 65 却承诺 vec4：结果恒正确的 UB | 搁置为已知盲区，写进 a2 README“判分覆盖不到”（62d59eb） |
| a2 等价写法（vin 不 pad、vout 第 0 维不 pad、方阵 tile 下 TM+1 与 TN+1 混用等） | 不是漏洞，无需处理 |
| 可选：a2 mutant m_sync_after_load、m_vout_pad_dim1_false | 未加：前者是竞态（R2 重跑 6 次里 1 次分数不同），违反“mutant 必须确定性失败”；后者越界的写可能落在别的 block 负责的位置，结果是否被正确值覆盖取决于 block 的执行顺序，无法保证确定性 |

复测正确写法（修改后的 test.cpp，8 张空闲卡并行）：a1 的 8 个正确变体（含 R2 目录中名为 m_flat_nonmerge_colmajor、m_trans_lowerswap_only 的两个实为正确写法）全部 11/11；a3 的 8 个 c_* 变体（1 warp 1 行、每 warp 4 行、1/16 wave、lane32/lane16、三路分派、两 tile 一轮）与 m_dispatch_mod8_ok_c、m_loop_from_0、x_loop_le、x_vo_nopad、x_v_rows_nopad 全部 15/15。

## R3 技术准确性审查

结论：错误 3、不严谨 6、建议 6。全部处理（f0d0fa6 讲义，e0ac0c7 / a224922 README 与骨架），改动的结论都在本机重新实测或对照了源码。

| 发现 | 处理 |
|---|---|
| E1 a3 问题 1 提示指向对齐 | 已修（同 R1 I-3） |
| E2 a1 README 源码路径 | 已修（同 R1 S-1） |
| E3 §1.1 hidden 维度定义错误 | 已修：hidden id 是 adaptor 对每一个维度（上、中、下层）的统一编号，上层/底层是其子集；图 1.1 中间两处标注改为“中间维度” |
| I1 §2.3 “pad 长度要与窗口一致” | 已修：有效性只和原长度比较，第二个参数只决定 pad 后的逻辑长度；引用 R3 的 (1, 1) 实验（复跑 /tmp/r3lab3/peek1，与 (8, 8) 输出相同） |
| I2 §3.1 `in_reduce_dims` 只支持 2D、归约第 1 维 | 已修：写明限制与第 186、237–249 行的 FIXME；`sequence<0>` 静默出错（复跑 red0：24024 88024 0 0 vs 28000…） |
| I3 “3 参数版本”的说法 | 已修：两个重载都是 4 个参数，改称“返回新累加器的版本 / 以已有累加器为第一个参数的版本”，内部调用位置改为第 276–291 行；a3 Hint 2 同步 |
| I4 图 3.2 xor 交换顺序画反 | 已修：^1、^2、^4；正文补“istage 从 0 递增” |
| I5 `move_tile_window` “只是加法” | 已修：沿变换链增量更新（`tensor_adaptor_coordinate.hpp` 第 172 行起），embed/pad 退化为加法，merge 有进借位 |
| I6 “最低 16 位段” | 已修：长度 16 的最低段，即 lane id 的低 4 位 |
| 建议 1 描述符还带向量承诺 | 已采纳：§1.1 补半句 |
| 建议 2 小部件 2.1 P 段顺序写死 | 已采纳：图注说明小部件只能表达按 X0、X1、R 排列的写法 |
| 建议 3 a3 骨架注释出处 | 已采纳：第 9 行、第 12–15 行与第 257–260 行；README 约束同步 |
| 建议 4 `block_sync_lds_direct_load` | 已采纳：等 vmcnt，gfx942 上 vmcnt 同时计读写，典型用途是 buffer_load 直接写 LDS；补行号 287–291 |
| 建议 5 store_tile 对 R 的处理 | 已采纳为线索（不给结论，因为它是 a3 书面问题 2）：§3.2 片段 ③ 指向 `replicate` 变换 |
| 建议 6 `warp_shuffle_down` 行号与越界行为 | 已采纳：行号 47–60；HIP `__shfl_down` 越界返回自己（amd_warp_functions.h 第 375–381 行），CK 版本按 64 回绕，本例不影响结果 |

## 早先延期的小问题（本批一并处理）

| 来源 | 处理 |
|---|---|
| a1 TODO a1-2 点名变换构造函数 | 已修：a1-2 TODO 列出四个 `make_*_transform`，a1-3/4/5 各自点名 |
| a1 测试加 M == N 的转置、TM == TN 的分块 | 已加（transposed 5×5，tiled (12, 8, 4, 4)） |
| a3 性能失败信息暗示判分检查向量化 | 已修：改为“检查访存模式与并行度；没有向量化是可能原因之一” |
| a3 加 ones(5, 260)（向量 4 的尾块） | 已加到 `many_col_tiles` |
| CLAUDE.md range 说明、只有内层维绕行、合并重复的 1-P 条目 | 已修 |
| Lab 2 讲义约第 587 行“没有比较指令” | 已修：“没有选择指令 `v_cndmask`” |

## 判分变化在文档里的同步

- a1 README：说明测试检查各维长度；用例表 transposed（三种形状）、tiled（两组参数）与各用例的“+ 各维长度”；“判分覆盖不到”首句。
- a3 README：用例表新增 `cols_mod4_eq2` 行、`many_col_tiles` 写出形状；约束补“至少留一个 Y”。
- a2 README：判分盲区补向量承诺一条。

## 已知并接受的取舍

- §3.3 的 scale_strip 与 §3.5 的分派函数仍是完整代码（逐元素例子，教窗口移动与模板分派），但第二轮修复后它们的分布、tile 形状、block 大小和分派条件都与行归约不兼容，见下节。
- a1 的“另一种写法”练习（§1.3）不给答案；小部件 1.1 仍按偏移显示五条链（README 已给出偏移公式，不泄露变换写法）。
- a2、a3 的性能用例都只是“不至于慢得离谱”的下限，不检查是否用 LDS / 是否向量化（README 已写明）。

## 修复第二轮（复审：a3 仍可按常量替换拼出）

复审者把 §3.1 的 `Enc8x64`（把 Y1 的 4 换成 kVec）放进 §3.3 scale_strip 的视图/窗口/循环（当时 tile 恰好是 8 × 16·kVec），加上 §3.2 的三个片段与 §3.5 的分派（block 128 → 64），得到 15/15。控制者裁定 R22：保留 a3 自主设计，打破这种巧合。

| 改动 | 内容 |
|---|---|
| §3.1 例子分布 | 列方向改为 Y1 4 在高段 × lane 16 在低段：推导、R 段、sync 的轮数与 `Enc8x64Reduced` 完全不变（static_assert 仍通过），但每个线程的 4 个元素相隔 16 列、不能向量化；正文写明这是为了演示，a3 的列方向、wavefront 数与 tile 大小要自己设计。图 3.1 与小部件预设“§3.1 归约前”同步，预设与 GPU `load_tile` 实测重新逐元素比对（0 处不一致） |
| §3.3 scale_strip | 分布改为行 = lane 高位 2、列 = warp 2 × lane 32 × Y kVec：tile 2 × 64·kVec、128 线程，一行横跨两个 wavefront、行方向没有 Y。新增 warn 框：逐元素运算可以这样分，拿去做行归约会同时违反两条约束（实测 `block_tile_reduce` 编译失败）。§3.4 的带宽实测按新 kernel 重测：承诺 4 约 3.3 TB/s（dwordx4）、承诺 2 约 2.5 TB/s、承诺 1 约 1.7 TB/s，拷贝约 2.1 TB/s |
| §3.5 分派 | 改为 `cols % 2` 二选一（V = 2 / 1），grid = ⌈rows / 2⌉；正文说明承诺 4 的条件和是否加档留给学生按 §3.4 推，grid/block 随分布重算 |
| 小问题 | a1 README / test.cpp 注释：长度错误影响的是 `pad_tensor_view`（它用 `get_length` 作原长度），tile window 不看长度；CLAUDE.md：带行间距的视图里小幅列越界落在行间空位，超过 ld − cols 才进入下一行 |

按常量替换重新拼装 a3（/tmp/fix17/asm）：

| 尝试 | 结果 |
|---|---|
| A：新 scale_strip 骨架（2 × 64·kVec、128 线程）+ §3.1 分布（Y1 换成 kVec）+ 三个片段 + §3.5 分派 | 能编译（窗口长度与分布不符 CK-Tile 不检查），0/15 |
| B：新 scale_strip 骨架连同它自己的分布 + 三个片段 | 编译失败（行方向没有 Y） |
| C：§3.1 分布原样，按 §2.1 清单从分布反推 tile 8 × 64、block 64 线程、grid ⌈rows/8⌉，再写循环 | 15/15，带宽约 2040 GB/s（比值约 0.97） |

结论：没有任何一段讲义代码能靠改常量直接拼成通过的 a3；能通过的唯一路径（C）需要学生自己从分布推出 tile 形状、block 大小与 grid，并且得到的是不能向量化的布局（每线程元素相隔 16 列），带宽低于 README 里“总是向量 1”的设计。它仍高于 0.80 的下限——这是 README“判分覆盖不到”已写明的盲区（性能用例不检查向量化），不是讲义泄题。

## 校准

所有比值 = 带宽 / 同进程实测 D2D 拷贝带宽（拷贝约 1990–2100 GB/s）。

### a2 tile-transpose（kCkTransposeRatio = 0.19，Task 15）

| 实现 | 比值 |
|---|---|
| 参考答案（64×64 tile、4 wave、LDS 行间距 TN+1、向量 4） | 0.54–0.57（S_min ≈ 0.54） |
| 64×64、LDS 不 padding | 0.52–0.54 |
| 32×32 tile（最慢的 LDS 设计） | 0.455–0.46 |
| 64×64、向量宽 2 | 0.65–0.67 |
| 64×64、交错行分布 | 0.55–0.57 |
| 跳过 LDS（跨步写全局内存） | ≈ 0.285（580–594 GB/s） |
| 候选慢 mutant m4（64 线程 block、1 个 P 维） | 0.51–0.52，S/N ≈ 1.04 < 2.25，按兜底规则删除 |

取值经过：兜底 floor2(S_min/1.67) = 0.32 → 为让 32×32 设计留 1.5× 降到 0.30 → 跳过 LDS 的正确写法实测约 0.285，会被非确定性地判挂 → 裁定 R20 改为 0.19（跳过 LDS 约 1.5×、参考答案约 2.8×、Lab 1 朴素转置约 0.06 低 3×）。性能用例因此只是下限，是否使用 LDS 由 README 与书面问题要求。

### a3 row-sum（kRowSumRatio = 0.80，Task 16）

比值 = 读带宽（只算读入字节）/ 拷贝带宽，所以可以大于 1。

| 实现 | 比值 |
|---|---|
| 参考答案（4 wave、每 block 8 行、向量 4） | 1.66–1.77 |
| 每 warp 1 行（TM = 4） | 1.43–1.52 |
| 每 warp 4 行（TM = 16） | 1.76–1.84 |
| 两个 tile 一次迭代 | 1.65–1.75 |
| 1 wave / block | 1.60–1.70 |
| 16 wave / block | 1.53–1.61 |
| 向量宽 2（cols % 2 分派，最慢的合理设计） | 1.22–1.35 |
| 候选慢 mutant m4（总是向量 1） | 1.04–1.13，S/N ≈ 1.47 < 2.25，按兜底规则删除 |

阈值 0.80：最慢的合理设计 1.22 / 0.80 ≈ 1.5×。标量版本也能通过，README“判分覆盖不到”已写明。本批新增的 `cols_mod4_eq2` 是正确性用例，不影响校准。

a1 无性能用例。

## 本批复测的其他实测

- 讲义片段：用 check.py 从 tutorial.html 自动抽取全部 16 个片段编译运行（0 warning）。§1：`cab` 与“另一种写法”逐坐标一致、长度 (4, 2, 3)；`k_b` / `k_b2` 偏移表与讲义一致（0 1 2 3 12 13 14 15 / 0 12 1 13 2 14 3 15）；`swizzled` 长度 (4, 8, 4)，偏移表与原表一致。§3.2 三个片段放进 1 wave/block 的 kernel：rows = 8、24、4096 行最大值逐位正确；全 1 求和 sync 前 64/64 个 lane 为 (4, 4)、sync 后 64/64 为 (64, 64)；输出窗口换用另一个 encoding 时编译报 `no matching function for call to 'store_tile'`，用相同 encoding 自建分布可以编译。其余 §2、§3 片段结果与讲义一致。13 处“读源码”摘录与 ROCm 7.2 源码逐行一致。
- 浏览器（Playwright，浅色 / 深色 / 390 px）：Lab 3 与 Lab 2 讲义 console 错误 0、无横向溢出；Lab 3 目录 22 项、3 张卡片；两个小部件的 7 个预设与 GPU 实测逐元素一致；图 1.3、图 3.2 截图核对。
- § 引用交叉检查：讲义、README、answers.md、骨架中所有 §x.y 都存在。

## 最终 `make verify`（第二轮修复后复跑，结果相同，a2 1115.7 GB/s、a3 3552.3 GB/s）

```
== a1-layout
   solution    满分 11/11
   mutant      m1_colmajor_swapped.hip          被抓住: col_major
   mutant      m2_transpose_wrong_dims.hip      被抓住: transposed_view
   mutant      m3_tiled_wrong_order.hip         被抓住: tiled
   mutant      m4_xor_dims.hip                  被抓住: xor_swizzled
   mutant      m5_transpose_lengths_wrong.hip   被抓住: transposed_view
   skeleton    1/11
== a2-tile-transpose
   solution    满分 12/12 bandwidth=1113.7GB/s
   mutant      m1_no_pad.hip                    被抓住: ragged
   mutant      m2_lds_read_same_layout.hip      被抓住: one_tile, multi_tile, ragged, large, bandwidth
   mutant      m3_out_origin_swapped.hip        被抓住: multi_tile, ragged, large, bandwidth
   skeleton    0/12
== a3-row-sum
   solution    满分 15/15 bandwidth_aligned=3511.8GB/s
   mutant      m1_no_sync.hip                   被抓住: aligned_small, many_col_tiles, ragged_cols, cols_mod4_eq2, ragged_rows, large, bandwidth_aligned
   mutant      m2_no_window_move.hip            被抓住: many_col_tiles, ragged_cols, cols_mod4_eq2, ragged_rows, large, bandwidth_aligned
   mutant      m3_vec4_always.hip               被抓住: ragged_cols, cols_mod4_eq2
   mutant      m4_dispatch_mod2_vec4.hip        被抓住: cols_mod4_eq2
   skeleton    0/15

VERIFY OK
```

Lab 2 只改了讲义一行文字，`make verify` 复跑仍为 VERIFY OK。harness 未改动，Lab 0 / Lab 1 无需重新 verify。

## R4 行文与术语审查

范围：Lab 3 讲义（`lab3-cktile-concepts/tutorial.html`）。四份独立审查：两份看表述与术语，两份看技术严谨性。第 1 轮 45 条（必须改 10）；第 2 轮复审 3 条（必须改 1）；第 3 轮四份审查均通过。

| 提交 | 内容 |
|---|---|
| e07a23c | 全文改写；术语表补列主序、行间距（ld）、`block_tile_reduce_sync` |
| e36ad5c | 第 2 轮：向量承诺随变换传递、除法优化的条件、§3.4 实验的 dword 屏蔽 |

| 发现 | 处理 |
|---|---|
| §1.3 两种写法“得到的描述符完全相同” | 已改：长度与偏移映射相同，C++ 类型不一定相同（折叠框） |
| §1.5 “unmerge 比 merge 便宜得多” | 已改：只说乘加合成与除法拆分；开销看汇编，2 的幂除法可变移位、其他常量仍要乘法加移位（第 2 轮） |
| §1.6 j ↦ j ^ i 是置换 | 已改：j ^ (i mod N)，行数大于列数时也成立 |
| §2.4 LDS 行间距 33 时承诺向量 4“两种写法都可以” | 已改：带 padding 的 LDS 视图如实写 1——奇数行首不对齐，转置读出视图最后一维 stride 也不是 1；本机结果正确只是碰巧，放进折叠框。与 a2 README 一致 |
| §3.1 写 `sequence<0>` 为何静默出错 | 已改：线程内合并写死按列，返回分布却按传入维度推导，两者对不上 |
| §3.2 一维输出“行数不整除时需要 pad” | 已改：连续一维输出越界由 buffer range 屏蔽，pad 可用但非必需 |
| §3.3 移动窗口“只根据增量更新” | 已改：依次调用 `update_lower_index`，merge 会对新坐标重新做魔数除法（源码核对，折叠框） |
| §3.4 “最内维”承诺；列数条件 | 已改：统一为“最后一维”，承诺随变换传递（第 2 轮）；列数条件还依赖每次访问从 V 的倍数列开始（折叠框） |
| §3.4 实验中第 2 行“整个向量被屏蔽” | 已改：按 dword 只屏蔽偏移 18、19，偏移 16、17 仍读到 24、25（第 2 轮） |
| §3.5 代码注释“cols 为偶数时才实例化” | 已改：两个版本都在编译期实例化，运行时只选启动哪个 |
| §4.1 “已拥有全部概念” | 已改：“大部分概念”，与表后一段一致 |
| §4.2 MFMA 32 × 32 × 8 未注明类型 | 已改：fp16 输入、fp32 累加；形状随数据类型而不同 |
| 表述：§3.2 sync 源码提示框一段塞五个知识点；§1.6 xor 谁打乱谁的说明 | 拆成编号要点；CK 与 HIP 的越界回绕差异放进折叠框 |
| 术语：hidden id、adaptor、embed、⊕、∏、“反向排他前缀积”、pitfall | 首次出现处补解释或换成文字 |
| 部分采纳 L3-010 | 小部件 2.1 的自检逻辑只看行方向有无 Y；小部件脚本按约束不改，在图注里提醒 `sweep_tile` 要求每个 X 维都有 Y |
| 部分采纳 L3-043 | 去掉未讲过的“坐标适配器”“物理坐标”，但不直接写出 R 维不参与地址计算：那是 a3 书面问题 2 的结论 |

留给用户决定的事项：

- 已处理：小部件 2.1（dist-explorer）的自检改为按讲义结论分别核对 `sweep_tile`（每个 X 维都要有 Y，§3.3）与沿列归约（行方向有 Y、一行不跨 wavefront，§3.1–§3.2；一维 tile 不适用）；预设 3 现在提示 X1 没有 Y、不能用 `sweep_tile`，预设 6 提示一维 tile 不再有列可归约。图注里的临时提醒相应改写。控件、预设与 id 未动，七个预设在浏览器中逐个核对，控制台无报错。
- 原有实测（各 peek 实验、redistribute、scale_strip 运行结果，承诺 4/2/1 的 3.3/2.5/1.7 TB/s，sync 与 xor_sync 的 `ds_bpermute` 条数，8 × 60 pad 求最大值得 0，跨 wavefront sync 得 64）沿用，本轮未重测；源码行号只核对了改动涉及的几处。

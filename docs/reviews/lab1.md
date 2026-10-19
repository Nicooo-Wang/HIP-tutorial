# Lab 1（lab1-hip-basics）审查记录

三份独立审查（提示词见 [PROMPTS.md](PROMPTS.md)，`{LAB}` = `lab1-hip-basics`），审查基线为提交 88fe4dd。
修复分两批：判分侧（Task 10 batch A）与文档侧（Task 10 batch B）。

| 提交 | 内容 |
|---|---|
| 51a995a | harness：哨兵区每侧 256 KiB、填带载荷的 NaN 字（越界读参与运算必然得 NaN）；性能计时改为 `hipDeviceSynchronize` 夹住的墙钟 |
| 16ad504 | a1：saxpy 换用 3 个 alpha、检查 x 未被修改与 x 哨兵；`add_large` n = 2²⁴ + 3 |
| 24f3056 | a2：性能用例在 CPU 上逐元素校验整个 8192² 输出 |
| d48a1dd | a3：大 n 全 1 精确用例（含 n%4≠0），整组重复 3 遍；性能用例随机 0/1 输入、精确比较 |
| 1724e25 | a4：性能用例用行和、列和覆盖全部元素，并在每个 32×32 输出块抽点 |
| b1ba374 | a3：阈值 0.40 → 0.28（让 kMaxBlocks=256 这种合理设计稳定通过） |
| 962f440 | Lab 1 讲义、README、TODO 注释按 R1/R3 修订；讲寄存器分块的实现细节 |

## R1 学生视角走查

结论：四个 Assignment 都一次通过（49/49），无阻塞。

| 发现 | 级别 | 处理 |
|---|---|---|
| README Hint 把讲义刻意不给的内容补回来：a3 Hint 1（树形第一层 stride/条件）、Hint 2（wave_reduce 完整循环）；a2 Hint 2（tile[y][x]/tile[x][y]、x=64） | 重要 | 已修：a1/a2/a3 的 Hint 2 改为引导问题与关键观察；a3 Hint 1 改为“用小部件数一数”；a4 的步骤提纲放进折叠的“Hint 3（卡住很久再看）”（962f440） |
| 讲义/README 性能数字过时、偏保守（学生 a4 测到 9503，a2 约 55%） | 重要 | 已修：统一写成“参考实现（简单设计）在空闲 MI308X 上实测约 …；更好的设计可以更高”，并给出 a2 ≈55–60%、a3 可超过拷贝带宽、a4 ≈9500 GFLOPS 的例子；数字取自 batch A 墙钟计时后的复测（962f440） |
| 寄存器分块的 C++ 实现细节（常量下标、展开，否则进 scratch）没讲 | 重要 | 已修：§5.4 新增“让累加器真的留在寄存器里”，含 `-Rpass-analysis=kernel-resource-usage` 的本机真实输出；§6 速查表加“性能不够 → 寄存器溢出到 scratch”一行（962f440） |
| a2 书面问题 3 里的 “mutant” 术语 | 小 | 已修：改为“判分不能保证每次都抓到这种 bug”（962f440） |
| a4 骨架没有 kernel 占位，没说要自己新增 | 小 | 已修：README 与 TODO(a4-2) 都写明要自己新增 `__global__` kernel（962f440） |
| a3 LDS 数组大小没有具体提示 | 小 | 已修：§4.3 注意框补“HIP 支持的 wavefront 宽度只有 32 和 64”（962f440） |
| a3 书面问题 4 讲义没讲 | 小 | 已修：题目标注“讲义没有直接讲，属于拓展”（962f440） |
| a1 书面问题 1 “何时值得”讲义没讨论 | 小 | 未改：可由 §2.4、§2.5 推出，答案开放 |
| 性能 FAIL 速查行对 a4 不够具体 | 小 | 已修：§6 该行补 §5.2/§5.4，并新增 scratch 一行（962f440） |
| a4 是绝对阈值、没有相对基准 | 小 | 已修：§5.4 说明为何用绝对阈值、繁忙时怎么办（962f440） |

## R2 判分对抗审查

结论：60 个错误实现中漏网 18 个；19 个正确实现无误判。

| 发现 | 处理 |
|---|---|
| S1 缓存上次调用结果（a2 w09、a3 w13、a4 w10） | 搁置：有意作弊，自学课程作弊只骗自己 |
| S1 私有 non-blocking 流 + 默认流 event 计时（a2 w11、a3 w14、a4 w12） | 计时漏洞已修（51a995a，墙钟夹住 `hipDeviceSynchronize`）：现在测到真实 kernel 时间，与参考答案相同、结果正确，所以满分是合理的。“忽略 stream 参数”本身在单 stream 测试下不可观测，搁置到讲 stream 时 |
| S1 按尺寸特判（a3 w12） | 搁置为作弊；顺带被随机 0/1 精确性能用例抓到（d48a1dd，11/14） |
| S2 a3 n≥2²² 时 float4 丢尾（w07） | 已修：大 n 全 1 精确用例 + 精确性能用例（d48a1dd），新增 mutant m5 |
| M1 竞争类偶发漏网：a3 树形缺每层屏障（w03） | 已加强：精确用例整组重复 3 遍后 10/10 次被抓（d48a1dd）；仍属竞争类，不作必挂 mutant；README 增加“判分覆盖不到的 bug”说明（962f440） |
| M1 a4 缺第二个屏障（WAR，w03） | 搁置：gfx942 上不可可靠观测（审查者 5 次 + 独立大用例均正确）。讲义 §5.3、§5.4 与 a4 README 明确告诉学生判分抓不到、必须自己推理（962f440） |
| M2 a2/a4 性能用例抽样过稀（a2 w10、a4 w11） | 已修：a2 全量校验（24f3056）、a4 行/列和 + 每块抽点（1724e25），新增 mutant a2 m4、a4 m6 |
| M3 a1 saxpy 写死 alpha、改写 x（w09、w07） | 已修（16ad504），新增 mutant m4、m5 |
| L1 a1 float 下标（w08） | 已修：`add_large` n = 2²⁴ + 3（16ad504），新增 mutant m6 |
| L2 a4 A 不补零（w07） | 已修：NaN 哨兵让越界读参与乘法必得 NaN（51a995a），新增 mutant m5 |
| L2 a2 LDS 越界读但写有边界检查（w06） | 未修：越界读到的值只落进不会写出的 tile 位置，且距离都在 256 KiB 哨兵内，无法确定性观测（详见 batch A 报告第 7 节） |
| L2 a4 累加器未初始化（w09） | 搁置：编译器恰好按 0 处理，不可观测；讲义 §5.4 与 a4 README 已提醒（962f440） |

## R3 技术准确性审查

结论：错误 2、不严谨 6、建议 11。文档侧全部处理（962f440），test.cpp 注释两处见末尾。

| 发现 | 处理 |
|---|---|
| E1 “普通 v_fma_f32 只有一半速率” | 已修：§1.3 改为两条路（packed，或 ≥2 wave/SIMD 双发射且操作数不读 SGPR），给出 26.5 / 28.3 / 14 TFLOPS 三个实测值，并点明 GEMM 内层不打包也能全速 |
| E2 `__shfl_down_sync(0xffffffff, …)` “能编译” | 已修：§4.3 拆成两件事——32 位掩码在 ROCm 7.2 编译失败（本机复现了 static_assert 报错），写死 16 的偏移能编译但结果错 |
| N1 “持续满负载时”约 1.39 GHz | 已修：“跑 kernel 时 1.39–1.42 GHz，与负载大小基本无关” |
| N2 Infinity Cache 与拷贝带宽变化的因果 | 已修：删掉因果括注，改成一般性说法 |
| N3 “D2D 2.2 TB/s” 暗示实践上限 | 已修：表格与说明改为拷贝 2.0–2.2、只读 3–3.6 TB/s（本次在空闲 GPU 4/6 复测），并说明拷贝带宽只是归一化尺子、不是上限 |
| N4 “一个 kernel 里做不到”/“必须两次启动” | 已修：§4.1 与 reduction.hip 第 5 行改为“最直接、最可靠的办法” |
| N5 a4 误差界称“宽松上界” | 已修：讲义 §5.4 与 a4 README（962f440）；test.cpp:12 注释在 fix round 1 改为“比典型舍入误差宽松得多、不是严格上界” |
| N6 校准注释算术 | a3 部分已由 batch A / b1ba374 重写；a2 test.cpp:33 在 fix round 1 改为“参考解约 0.31–0.35、naive 约 0.062–0.064，余量参考 ≥2.2×、naive ≈2.23×” |
| 建议 1 MFMA 频率折扣 | 已修：§5.5 补“本机按约 1.4 GHz 实测 bf16 MFMA 约 230 TFLOPS” |
| 建议 2 LDS 两半模型实测确认；b64/b128 规则不同 | 已修：§3.6 补区分实验与 ds_read_b64/b128 提示 |
| 建议 3 “正好 2 条 cache line”需 128 B 对齐 | 已修 |
| 建议 4 8192 B 不等于 DRAM 流量 | 已修：改为“64 次 cache line 级访问” |
| 建议 5 指针必须指向显存 | 已修：“GPU 能访问的内存” |
| 建议 6 LDS 超限编译报错仅限静态声明 | 已修 |
| 建议 7 VGPR_Count 是分配量 | 已修 |
| 建议 8 stride ≤ 32 → ≤ 64 | 已修 |
| 建议 9 `__shfl_down` width 语义 | 已修 |
| 建议 10 fill_kernel int 溢出 | 已修：§2.5 加一条细节说明 |
| 建议 11 a4 README 漏 40×70×33；HBM “4 倍”像推导 | 已修 |

## 判分变化在文档里的同步

- 哨兵：Lab 1 §2.3、§6 速查表、a1–a4 README 的“得到 nan”都补上“也可能是越界读到 NaN 哨兵”。
- a1 `add_large` 尺寸 2²⁴ + 3（§2.4 两处）。
- a2 性能用例逐元素校验整个输出（§3.7 已由 batch A 改；README 补充）。
- a3 阈值 28%（§4.1 注、README 用例表）。§4.1 原来的“block 数只要能把 80 个 CU 填满（几百个 block）就够了”与实测矛盾（256 个 block ≈ 40%、1024 个 ≈ 67%），已改为：ws 大小是硬上限；光填满 CU 不够，每个 SIMD 需要多个驻留 wave 掩盖延迟（256 × 4 wave ÷ 320 SIMD ≈ 3 wave/SIMD 太少），并指向实测数字（fix round 1）。
- §5.4 原文“寄存器不能按运行时下标寻址”在 gfx942 上不成立（复审者编译 `acc[(idx+k)&15]` 得到 38 VGPR、ScratchSize 0、`s_set_gpr_idx_on/off`），已改为“代价高、编译器通常避免，下标/边界编译期未知时数组常常落进 scratch”（fix round 1）。

## 已知并接受的取舍

- Lab 1 §2.4 仍写出向上取整 `(n + B - 1) / B`：它是对 Lab 0 的复习（Lab 0 §3.3 已改为引导学生自己推导），a2 骨架的 launch 也本来就用它，所以 Lab 1 a1-2 实际上不考这个。
- Lab 1 §3.4、§3.6 讲解了 a2 的大部分设计（x 维 64、按列从 LDS 读、padding）：a2 是难度 2（“补全设计”），讲义负责把技巧讲清楚，README Hint 只给引导问题，不再重复设计。
- a4 性能用例的全元素校验写进 README 用例表。

## 校准

所有数字来自空闲 MI308X（gfx942），同进程 D2D 拷贝带宽为基准。“事件计时”是最初校准（Task 6–8），“墙钟”是 batch A 改计时后的复测。

### a2 transpose（kLdsRatio = 0.14）

| 实现 | 事件计时 | 墙钟 |
|---|---|---|
| 拷贝带宽 | 1937–1994 GB/s | 约 1860–2110 GB/s |
| 参考答案（64×64 tile、block 64×4、padding） | 660–662 GB/s，S = 0.334–0.341 | 669–670 GB/s，S = 0.313–0.350 |
| 慢 mutant m3（LDS 版调用 naive） | 123–125 GB/s，N = 0.062–0.064 | 133–134 GB/s，N = 0.0621–0.0627 |
| 余量（墙钟） | | 参考 ≥ 2.24×，naive ≥ 2.23× |
| 其他正确设计（R2） | tile32 / block 32×8 + pad 1185–1230 GB/s（约 55–60%）；tile64 / 16×16 每线程 4×4 约 716；tile32 无 pad / 32×32 约 873 | |
| 参考答案去掉 padding（本批，GPU 5） | 677.5–679.6 GB/s（有 padding 659.8–660.1）——瓶颈在显存，去掉 padding 不变慢；README 书面问题 2 已提醒“差别可能不大，如实记录” | |

### a3 reduction（kReduceRatio：0.03 → 0.45 → 0.40 → 0.28）

| 实现 | 事件计时 | 墙钟 |
|---|---|---|
| 拷贝带宽 | 1950–1971 GB/s | 约 2050–2130 GB/s |
| 参考答案 | 1477–1483 GB/s，S ≈ 0.754 | 1375–1392 GB/s，S = 0.656–0.676 |
| 慢 mutant m3（单 block） | 3.5 GB/s，N ≈ 0.0018 | 3.5 GB/s，N ≈ 0.0015 |
| kMaxBlocks = 256 的参考结构（正确） | 830 GB/s | 793 GB/s，约 0.38–0.41 |
| 其他正确设计 | block 512 / 320 blocks 约 0.92；LDS 树 + 每 block atomicAdd 约 0.74；R2 c01–c07 1292–2434 GB/s | |

取值经过：Task 7 按 √(S·N) 得 0.03（太宽松）→ 控制者裁定 floor2(S/1.67) = 0.45 → 墙钟计时后 S 降到约 0.67，按同规则取 0.40 → batch A 复审发现 kMaxBlocks=256 的合理设计（≈0.40）在 0.40 处成了“抛硬币”，改为 0.28（b1ba374）。0.28 时：参考答案最差一次 2.3× 余量，256-block 变体最差一次约 1.34×（平均约 1.4×），单 block 版低于阈值约 190×。

### a4 gemm（原 kTiledGflops = 2000，已改为 kTiledRatio = 0.065 × 同进程 FMA 吞吐）

| 实现 | 事件计时 | 墙钟 |
|---|---|---|
| 参考答案（32×32 tile、32×8 block、每线程 4 输出、padding） | 4635.6–4639.7 | 4626–4630 |
| 慢 mutant m3（tiled 调用 naive） | 1147.7–1148.8 | 1146.1–1146.3 |
| 余量（墙钟） | | 参考 2.31×，naive 1.745× |
| 其他正确设计 | 16×16 tile、每线程 1 输出 3106–3108；32×32 无 padding 4549–4553；R2：64×64×16 / 每线程 4×4 9956–9990；无 LDS 4×1 寄存器分块 1529（不过线，符合设计） | |

最初按 √(S·N) 取 2300，控制者为照顾 16×16 设计（余量 1.35×）下调为 2000。

**第一期终审修复（控制者裁定 R24）**：绝对阈值在同卡有 3 个并发参考答案进程时会误判正确代码（审查者实测参考答案 4630 → 2443）。改为相对同进程实测 fp32 FMA 吞吐：`hk::fma_peak_gflops()`（harness 新增，纯寄存器 `v_pk_fma_f32` 微基准，1280 × 256 线程、约 1.6 ms，不含任何 GEMM 结构，不泄露 a4-1 答案），与学生 kernel 交替计时 10 轮、各取最优，阈值 = 0.065 × 微基准吞吐。

| 实现 | 空闲 GPU 2（3 次） | 空闲 GPU 5（3 次） | 比值范围 |
|---|---|---|---|
| FMA 微基准 | 26762–26807 GFLOPS | 26791–26824 GFLOPS | — |
| 参考答案 | 4629–4631 | 4635–4636 | 0.1728–0.1730 |
| 16×16 tile、每线程 1 输出（R2 c01） | 3081–3083 | 3086–3087 | 0.1150–0.1152 |
| 32×32 tile、32×32 线程 block、无 padding（R2 c03） | 3091–3092 | 3092–3093 | 0.1152–0.1155 |
| 64×64、每线程 4×4（R2 c02） | 9952–9972 | 9974–9976 | 0.372 |
| 无 LDS、寄存器 2×2（R2 c04） | 1523–1527 | 1519–1527 | 0.057（不过线，与原绝对阈值下结论相同） |
| naive（m3） | 1146–1147 | 1146–1147 | 0.0428–0.0429 |

取 0.065（√ 规则区间：naive ≥1.5× 要求 ≥0.0643，16×16 ≥1.5× 要求 ≤0.0767，取靠下的一端以利抗干扰）：naive 低于阈值 1.52×，16×16 变体高于 1.77×，参考答案 2.66×；空闲时约合 1740 GFLOPS。

抗干扰（GPU 2/5，同一张卡上另有进程；“持续发射”是 2–3 个进程各自循环 `launch + hipDeviceSynchronize` 参考 GEMM，比审查者的场景重得多）：

| 干扰 | 参考答案 | 16×16 变体 | naive（m3） |
|---|---|---|---|
| 2 个本用例副本循环运行（审查者场景） | 0.1727–0.1729，全过 | 0.071–0.115，全过 | 0.0427–0.0428，全挂 |
| 3 个本用例副本循环运行 | 0.1727–0.1729，全过 | 0.115–0.135，全过 | 0.0426–0.0428，全挂 |
| 2 个进程持续发射参考 GEMM（8 次） | 0.078–0.108，全过（约 1300–1600 GFLOPS，原绝对阈值下全挂） | 0.060–0.084，8 中 7 过 | 0.038–0.048，全挂 |
| 3 个进程持续发射参考 GEMM（13 次） | 0.083–0.110，全过 | 0.062–0.087，13 中 10 过 | 0.035–0.052，全挂 |

持续满载时 GEMM 类 kernel 掉速（约 3.5×）比 FMA 微基准（约 2×）多，所以比值会下降；16×16 变体在持续满载下单次约 1/5 落到阈值下（21 次中 4 次，其中 3 次连续出现在同一段负载里，说明失败有相关性），判分器 3 次重试能救回大部分，但不能保证；失败信息与性能用例前的提醒会显示 GPU 占用，提示换卡。参考答案在所有场景下最低 0.078（1.2× 余量）；naive 在所有场景下最高 0.052（低于阈值 1.25×），即不会因负载而误判通过。用 32×32 = 1024 线程 block 的设计（c03）在持续满载下几乎拿不到 CU（约 170–400 GFLOPS，比 naive 还慢），任何基准都救不回来；空闲或间歇负载下正常通过。也试过别的基准：从 LDS 读操作数的 FMA 循环满载下同样只掉约 2×，没有更好；每 block 占 16 KiB LDS 的 FMA 基准满载下掉约 4.5×（比 GEMM 还多），会让 naive 的比值升到 0.09–0.10、越过阈值误判通过，违反“不放过错误代码”，未采用。


### 其他实测（本批）

- 拷贝 / 只读带宽（GPU 4、6，墙钟，best of 20）：256 MiB 拷贝 2072–2119 GB/s、只读 3068–3141 GB/s；1 GiB 拷贝 1883–1970 GB/s、只读 3537–3625 GB/s。
- `-Rpass-analysis=kernel-resource-usage`：常量边界 + `#pragma unroll` 的 4×4 累加器 VGPRs 22、ScratchSize 0；运行时边界的同一数组 VGPRs 10、ScratchSize 80 bytes/lane。参考 a4 tiled kernel：VGPRs 84、ScratchSize 0、Occupancy 5、LDS 8448 B。
- `__shfl_down_sync(0xffffffff, …)` 在 ROCm 7.2 编译失败：`static assertion failed due to requirement 'sizeof(unsigned int) == 8': The mask must be a 64-bit integer.`

## 最终 `make verify`（962f440，含 b1ba374 的 0.28 阈值）

```
== a1-vector-add
   solution    满分 11/11
   mutant      m1_no_bounds.hip                 被抓住: add_small, add_ragged, add_large
   mutant      m2_no_stride.hip                 被抓住: saxpy_needs_stride
   mutant      m3_alpha_ignored.hip             被抓住: saxpy_fits_grid, saxpy_small, saxpy_needs_stride
   mutant      m4_saxpy_clobbers_x.hip          被抓住: saxpy_fits_grid, saxpy_small, saxpy_needs_stride
   mutant      m5_saxpy_hardcoded_alpha.hip     被抓住: saxpy_fits_grid, saxpy_small, saxpy_needs_stride
   mutant      m6_float_index.hip               被抓住: add_large
   skeleton    0/11
== a2-transpose
   solution    满分 10/10 lds_bandwidth=659.4GB/s
   mutant      m1_wrong_leading_dim.hip         被抓住: naive_all_shapes
   mutant      m2_lds_no_bounds.hip             被抓住: lds_all_shapes
   mutant      m3_lds_is_naive.hip              被抓住: lds_bandwidth
   mutant      m4_lds_skip_last_row_large.hip   被抓住: lds_bandwidth
   skeleton    0/10
== a3-reduction
   solution    满分 14/14 shfl_bandwidth=1389.1GB/s
   mutant      m1_drop_tail.hip                 被抓住: lds_exact_ones, lds_random, shfl_exact_ones, shfl_random, shfl_bandwidth
   mutant      m2_atomic_no_init.hip            被抓住: lds_exact_ones, lds_random, shfl_exact_ones, shfl_random, repeatable_output, shfl_bandwidth
   mutant      m3_single_block.hip              被抓住: shfl_bandwidth
   mutant      m4_shfl_width32.hip              被抓住: shfl_exact_ones, shfl_random, repeatable_output, shfl_bandwidth
   mutant      m5_float4_drops_tail_large.hip   被抓住: lds_exact_ones, shfl_exact_ones, shfl_bandwidth
   skeleton    0/14
== a4-gemm
   solution    满分 14/14 tiled_gflops=4623.7GFLOPS
   mutant      m1_naive_b_colmajor.hip          被抓住: naive_small, naive_all_shapes
   mutant      m2_tiled_drop_k_tail.hip         被抓住: tiled_all_shapes
   mutant      m3_tiled_is_naive.hip            被抓住: tiled_gflops
   mutant      m4_tiled_no_bounds_load.hip      被抓住: tiled_all_shapes
   mutant      m5_tiled_A_not_zero_padded.hip   被抓住: tiled_all_shapes
   mutant      m6_tiled_skip_last_tile_large.hip 被抓住: tiled_gflops
   skeleton    0/14

VERIFY OK
```

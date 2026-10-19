# 第一期（基础设施 + Lab 0–3）收尾总结

日期：2026-10-16　基线：提交 d86e130（Lab 3 审查修复之后）

## 全量验证（`make verify-all`）

在仓库根目录运行 `make verify-all`，退出码 0。机器 8 张 MI308X 当时均空闲（`rocm-smi` GPU use 0%），`verify.py` 自动选卡。

| 阶段 | 结果 |
|---|---|
| `common/test_grade.py`（判分基础设施自测） | 10 个测试全部 ok（51.6 s） |
| lab0-env | VERIFY OK |
| lab1-hip-basics | VERIFY OK |
| lab2-cktile-intro | VERIFY OK |
| lab3-cktile-concepts | VERIFY OK |

### 每个 Assignment

| Lab | Assignment | 参考答案 | mutant（全部被抓住） | 骨架 |
|---|---|---|---|---|
| 0 | a1-hello | 6/6 | 2 | 0/6 |
| 0 | a2-device-query | 6/6 | 2 | 0/6 |
| 1 | a1-vector-add | 11/11 | 6 | 0/11 |
| 1 | a2-transpose | 10/10 | 4 | 0/10 |
| 1 | a3-reduction | 14/14 | 5 | 0/14 |
| 1 | a4-gemm | 14/14 | 6 | 0/14 |
| 2 | a1-cpp-meta | 13/13 | 6 | 0/13 |
| 2 | a2-ck-vector-add | 12/12 | 5 | 0/12 |
| 3 | a1-layout | 11/11 | 5 | 1/11 |
| 3 | a2-tile-transpose | 12/12 | 3 | 0/12 |
| 3 | a3-row-sum | 15/15 | 4 | 0/15 |

共 11 个 Assignment、48 个 mutant，每个 mutant 至少挂一项；骨架均不满分（Lab 3 a1 骨架 1/11：`make_row_major` 作为示例已给出，对应用例自然通过）。

### 性能用例

“verify 实测”是 `make verify-all` 中参考答案的读数；“复测”是随后在 GPU 0 上用同一参考答案直接运行该用例 3 次（`<binary> <case>`）的读数，拷贝带宽为同一进程内 `hipMemcpyDeviceToDevice` 实测值。Lab 1 a2/a3 的用例通过时不打印拷贝带宽，表中拷贝带宽取自单独程序以相同字节数调用 `hk::copy_bandwidth_gbs` 的 3 次读数（约 1850–2070 GB/s），比值为估算。

| 用例 | 阈值 | verify 实测 | 复测（3 次） | 参考答案 / 拷贝带宽 | 余量（参考 ÷ 阈值） |
|---|---|---|---|---|---|
| Lab 1 a2 `lds_bandwidth` | ≥ 14% × 拷贝带宽 | 658.7 GB/s | 658.8 / 658.2 / 658.3 GB/s | ≈ 0.32–0.36 | ≈ 2.3–2.5× |
| Lab 1 a3 `shfl_bandwidth` | ≥ 28% × 拷贝带宽 | 1382.1 GB/s | 1388.1 / 1381.6 / 1387.6 GB/s | ≈ 0.68–0.74 | ≈ 2.4–2.6× |
| Lab 1 a4 `tiled_gflops` | ≥ 2000 GFLOPS（绝对） | 4622.4 GFLOPS | 4630.1 / 4627.9 / 4620.8 GFLOPS | — | 2.3× |
| Lab 2 a2 `bandwidth` | ≥ 55% × 拷贝带宽 | 2967.9 GB/s | 3422.9 / 3424.1 / 3317.7 GB/s（拷贝 2086 / 2090 / 2073） | 1.60–1.64 | ≈ 2.9× |
| Lab 3 a2 `bandwidth` | ≥ 19% × 拷贝带宽 | 1114.2 GB/s | 1114.7 / 1117.4 / 1116.4 GB/s（拷贝 1996 / 1998 / 1993） | 0.56 | ≈ 2.9× |
| Lab 3 a3 `bandwidth_aligned` | ≥ 80% × 拷贝带宽 | 3486.5 GB/s | 3545.9 / 3553.0 / 3497.3 GB/s（拷贝 2060 / 2073 / 1993） | 1.71–1.75 | ≈ 2.1–2.2× |

六个性能用例的参考答案都比阈值高 2 倍以上，与各 Lab 审查时的校准一致（慢 mutant 一侧的余量见 `lab1.md`–`lab3.md`）。

观察（未改动）：Lab 2 a2 README 写“参考答案约为拷贝基准的 1.45 倍”，本次复测为 1.60–1.64（verify 内那次约 1.4–1.5，取决于同进程拷贝读数）。“约”的说法不影响判分，留给后续统一更新性能数字时处理。

## 本次收尾的其他改动

- 根目录 `Makefile`：`make test-infra`、`make verify-all`。
- `README.md`：课程目标与对象、Lab 0–8 路线图（第一期已完成）、快速开始、学习流程、判分说明、solutions 使用约定、共享 GPU 说明、目录结构、维护者说明（含新增 Lab 的步骤）。
- `site/index.html`：Lab 0、Lab 3 卡片描述按讲义实际内容更新；页脚指向根目录 README。用 Playwright（Node）在浅色/深色 × 1280/390 px 下逐个点击 Lab 0–3 卡片：标题分别为“Lab 0 · 环境与工具”“Lab 1 · HIP 基础”“Lab 2 · CK-Tile 入门”“Lab 3 · CK-Tile 核心概念”，均无 console error / 请求失败；390 px 下无横向滚动（scrollWidth = clientWidth = 390）。

## 跨 Lab 连续性审查

按 Lab 0 → 3 的顺序通读讲义、README、骨架与 answers.md，只看跨 Lab 的衔接，共 26 条：断层/矛盾 4、前向引用 7、术语不一致 7、小问题 8。全部已处理（F2 以“明确推迟到 Lab 4”的方式处理），§ 编号未变。

**断层 / 矛盾**
- C1 Lab 3 §3.2 说学生在 Lab 1 a3 写的是 `for (d = 32; …)`，与 Lab 1 §4.3“用 warpSize 推出”矛盾 → 改为“和你在 Lab 1 a3 写的 `wave_reduce` 同一个树形结构（delta 从 warpSize 推出）”，不再给出代码。
- C2 Lab 2/3 写死 64 与 Lab 1 warpSize 原则未对上 → Lab 2 §3.2 首次出现处加说明框：warpSize 是运行期值，而分布是编译期类型；CK-Tile 用 `constexpr get_warp_size()`（`core/arch/arch.hpp` 第 63–70 行，gfx9 返回 64，`get_warp_id` 在第 106 行用它），只编 gfx942 所以 64 确定；运行期用 warpSize、编译期用 `get_warp_size()` 或 64 + `static_assert`（已在 gfx942/gfx1100 上验证 static_assert 行为）。正文 `/ 64` 改为 `/ get_warp_size()`，Lab 2/3 清单的“= 64”注明即 `get_warp_size()`。kernel 代码未改。
- C3 README“每个 Lab 的收尾作业是自主设计”不实 → 改为逐 Lab 说明：Lab 0 填空；Lab 1 填函数体 → 补全设计（a2）→ 自主设计（a4）；Lab 2 回到填空；Lab 3 填空 → 补全设计（a2）→ 自主设计（a3）。
- C4 Lab 3 a2 书面问题 3 依赖 Lab 1 参考答案内部细节 → 改为与“你自己的 Lab 1 a2”比较（在 lab1 下 `make run A=a2 CASE=lds_bandwidth`，借本作业同时段的拷贝带宽换算比值），Lab 1 §3.7 的 0.31–0.35 只作参照；answers.md 模板同步加一行。

**前向引用**
- F1 `--save-temps` 从未讲过 → Lab 2 §2.3 首次使用处加说明框：完整命令、在临时目录生成、`<源名>-hip-amdgcn-amd-amdhsa-gfx942.s`、按 kernel 名搜修饰名、到 `s_endpgm` 为止、`.bc` 用 llvm-dis；作业文件无 kernel 时如何改 `make run` 背后的命令（实测文件名 `test-hip-…-gfx942.s`）。Lab 2 a1 README 指向这里。
- F2 Lab 1 承诺的 b64/b128 bank 分组未兑现 → Lab 1 §3.6 改为“Lab 3 §1.6 会再提，具体规则留到 Lab 4”；Lab 3 §1.6 加说明框：本节与小部件按 4 字节/半 wavefront 模型，按 4 个一组 swizzle 只保证向量访问不被拆开、行首对齐不变，b128 下是否无冲突留到 Lab 4 按其分批规则重算，本 Lab 作业用不到。未在本期给出 b128 分批规则（未经本机实测，不写）。
- F3 `__device__` 函数与模板 kernel 未讲 → Lab 1 §2.2 补第五点；a3 骨架头注释指向 §2.2。
- F4 编译失败的判分输出未展示 → Lab 0 §1.3 加第三种失败信息“aK: 编译失败（0 分）”+ 从第一条 `error:` 读起。
- F5 Lab 3 a1 README 用了未讲的 `make_naive_tensor_descriptor_packed` → 改为讲义讲过的一维 naive descriptor 写法。
- F6 Lab 3 a3 没说 kernel 写在哪 → 骨架加 `// >>> 你的 kernel（及辅助函数）<<<` 标记与说明（文件作用域、`launch_ck_row_sum` 之前），README 同步；根 README 学习流程补“自主设计题在标记之外新增函数，位置见该题 README”。
- F7 原子操作从未讲过 → Lab 1 §4.5 加 `atomicAdd(float*, float)` 最小说明（返回旧值、同址串行化、顺序不定）；a3 书面问题 4 指向它。

**术语不一致**
- T1 warp 含义 → Lab 0 术语表注明 CK-Tile 的 warp = wavefront（64 lane）；Lab 2 §3.2 说明框点明 `get_warp_id()` 即 Lab 1 的 `wave_id`。
- T2 block/workgroup、跨 CU → README“每个 block（AMD 文档叫 workgroup）”、Lab 8“跨 block 同步”；site 同步（LDS 64 KiB / block、跨 block 同步）。
- T3 OK vs PASS → 全部改为“全部 OK”；Lab 0 §1.4 说明 `make run` 每用例一行 OK/FAIL、汇总行“通过 x/y”、不自动挑卡。
- T4 “判分覆盖不到的 bug”格式 → 统一为独立 `## 判分覆盖不到的 bug`，放在 Hints 与书面问题之间（Lab 1 a2/a3/a4 由引用块改为独立节，Lab 2 a1 补“bug”并移位）。
- T5 Lab 2 标题 → `<title>`、页头、目录标题改为“C++ 速成 + CK-Tile 入门”。
- T6 带宽表示法 → Lab 1 §1.3 定义“比值 = 你的带宽 ÷ 同进程拷贝带宽”（14% 即 0.14）；Lab 2 a2、Lab 3 a2/a3 README 的阈值与参考数字统一写成比值。
- T7 Lab 3 a3 Hint 2 的“三层” → 改为“三步（线程内 → 跨 lane → 写回）”。

**小问题**
- M1 页脚 → 四份讲义统一“上一课：Lab N 标题 · 下一课：Lab N+1 标题”并加链接（Lab 3 下一课为 Lab 4（第二期），无链接）。
- M2 Lab 2 §4 无回指 → Lab 3 §2.3 加“回头答 Lab 2 §4.1 第 1 问 / §4.3 第 2 问”，§3.3 末尾指向 Lab 2 §4.3 第 3 问。
- M3 性能用例自动重试 → Lab 1 §6 失败速查表加一行（最多 3 次、有一次通过即通过、附 GPU 占用、`make run` 不重试）。
- M4 Lab 3 两个 README 说通过时打印比值 → 改为 `make run` 才打印，`make grade` 通过时只显示带宽。
- M5 Lab 2 “64 个哨兵与 Lab 1 相同” → 改为“原理同 Lab 0 §1.3，harness 是 256 KiB，这里只留 64 个元素”。
- M6 Lab 0 暴露 `make verify` → 删去。
- M7 README 示例命令只在 Lab 1 有效 → 改为 `CASE=<用例名>`，并注明 Lab 1 的例子。
- M8 Lab 2/3 未提 profiler → Lab 2 §3.4 末尾加“Lab 1 §6 的工具照常可用”，说明 rocprofv3 trace 里 kernel 名（含模板参数，实测）。

验证：四份讲义 + site 浅色/深色/390 px 宽无控制台错误、无横向溢出、跨页锚点全部存在；全部 § 交叉引用（535 处）复核；新增 static_assert 片段与 `--save-temps` 命令本机编译通过；改动后重跑 `make verify-all`：test_grade 通过，4 个 Lab 均 VERIFY OK（solutions 全过、mutant 全被抓、骨架不满分；改过的 Lab 1 a3、Lab 3 a3 骨架仍能编译且 0 分）。

# HIP / CK-Tile 学习仓库设计文档

日期：2026-10-15　状态：待审阅

## 1. 目标与背景

- 学习者将在 AMD Instinct MI308X（gfx942 / CDNA3）上做 mega kernel 方向的研究，需要系统掌握 HIP 与 CK-Tile kernel 编写。
- 学习者现状：了解 GPU 编程大致概念，未系统写过 kernel；会用 C++ 但模板元编程不熟；中文交流。
- 形式：Tutorial（讲解）+ Assignment（动手）+ 自动判分（验证），参考 MIT 6.828/6.S081 的"讲义配 lab、明确任务、Hints、书面问题、`make grade`"思想，但不照搬。
- 成功标准：学习者只靠 Tutorial + Hints 能独立完成 Assignment；判分不误判对的、不放过错的；每个 Lab 经过独立审查。

### 第一优先级：学习连续性
每个 Assignment 只引入少量新概念，且必须在此前的 Tutorial 中讲过；后续 Assignment 复用前面写过的代码/概念。

## 2. 分期计划

总体大纲一次定好，实现分期进行。**第一期：公共基础设施 + Lab 0–3**，用户试用反馈后再做后续 Lab。

| Lab | 主题 | Assignments（初拟，实现时可调整） | 期 |
|---|---|---|---|
| 0 环境与工具 | 编译、运行、判分流程、rocprof 入门 | hello kernel；查询设备属性 | 1 |
| 1 HIP 基础 | 线程层级、内存层级、同步 | vector add → 2D 转置（naive → LDS + 消除 bank conflict）→ reduction（LDS 树形 → wavefront shuffle）→ 朴素 GEMM | 1 |
| 2 C++ 速成 + CK-Tile 入门 | `number<N>`、`sequence`、`tuple`、`static_for`、lambda、`if constexpr`；CK-Tile 写 vector add | C++ 小练习 → CK-Tile elementwise | 1 |
| 3 CK-Tile 核心概念 | tensor descriptor / coordinate transform（layout）、tensor view、tile window、tile distribution、load/store_tile、sweep_tile | 手写 layout 变换 → tile window 转置 → 行归约 | 1 |
| 4 MFMA 与 tiled GEMM | MFMA 指令、HIP 手写分块 GEMM、CK-Tile warp/block GEMM 组装 | 带性能门槛 | 2 |
| 5 GEMM 进阶 | pipeline、double buffer、epilogue 融合、Grouped GEMM | 带性能门槛 | 2 |
| 6 多卡通信基础 | P2P / IPC、RCCL、手写 all-to-all（device 端 flag 同步） | | 3 |
| 7 通算融合 | Grouped GEMM + all-to-all 融合（MoE 场景），后端实现剖析 | | 3 |
| 8 Mega kernel 技术 | persistent kernel、device 端任务队列与调度、跨 CU 同步、指令级流水、主流实现案例 | | 4 |

## 3. 仓库结构

```
HIP-tutorial/
├── CLAUDE.md                 # 项目约定（供后续会话遵守）
├── README.md                 # 课程总览、学习路线、快速开始
├── common/
│   ├── harness.hpp           # HIP_CHECK、随机数据、容差比较、哨兵越界检测、计时、JSON 输出
│   ├── grade.py              # 判分驱动（仅 Python 标准库）
│   ├── verify.py             # 用 solutions 与 mutants 校验判分器
│   └── tutorial.css / tutorial.js
├── site/index.html           # 讲义总目录
├── labN-<name>/
│   ├── tutorial.html
│   ├── Makefile              # make grade / make grade-aK / make run-aK / make verify
│   ├── aK-<name>/
│   │   ├── <name>.hip        # 学生编辑的文件（挖空）
│   │   ├── test.cpp          # 判分测试（学生不改）
│   │   ├── README.md         # 任务、Hints（分级）、书面问题
│   │   └── mutants/          # 故意写错的变体，verify 时必须 FAIL
│   └── answers.md            # 书面问题作答（不判分）
└── solutions/labN-<name>/aK-<name>/<name>.hip   # 参考答案（仓库内，用户自律不偷看）
```

## 4. Assignment 设计

### 4.1 挖空的三个难度（随课程推进逐步过渡）
1. **填函数体**：签名与 launch 配置给定，只空 kernel 关键几行。
2. **补全设计**：给接口与设计提示，block 尺寸、LDS 布局、循环结构由学生决定。
3. **自主设计**：只给 host 接口与测试，kernel 从头写（用于每个 Lab 的收尾 Assignment）。

### 4.2 引导
- 每个空用 `// TODO(aK-n): <要做什么>` 标注，附讲义对应小节锚点与常见坑。
- README 中 Hints 分级：Hint 1 指方向，Hint 2 给关键步骤，不直接给代码。
- 书面问题（6.828 风格）引导思考"为什么"，写在 `answers.md`，不判分。

### 4.3 学生文件与测试的接口
学生文件只暴露约定好的 host 函数（如 `void launch_vector_add(const float*, const float*, float*, int n, hipStream_t)`）；`test.cpp` 包含/链接该文件并调用，学生可自由改变内部实现（尤其难度 2、3）。

## 5. 判分

### 5.1 流程（`make grade`）
1. 用 `/opt/rocm/bin/hipcc --offload-arch=gfx942` 编译；编译失败该 Assignment 记 0 分并显示错误。
2. 选卡：读取 `rocm-smi` 的 GPU 使用率与显存占用，选最空闲的一张；可用 `HIP_VISIBLE_DEVICES` 覆盖。
3. 运行测试程序（带超时），解析其 JSON 输出。
4. 输出 6.828 风格结果：每项 `OK`/`FAIL`、得分、性能行，最后 `Score: x/y`。

### 5.2 正确性
- 多组尺寸：极小、非整除（如 n=1、1023、4097）、较大；多个随机种子。
- 与 CPU 参考实现比较，容差按数据类型设定（fp32 / fp16 / bf16 不同；累加类用相对+绝对容差）。
- 输出缓冲区前后放哨兵值，检测越界写；输入缓冲区校验未被修改。
- 失败时打印首个错误位置、期望值与实际值，便于调试。

### 5.3 性能（计入分数，从需要性能的 Assignment 开始，主要是 Lab 4 起）
- warmup 后多次运行取最好成绩；阈值相对参考实现设得宽松（例如 ≥50%）。
- 测量前检查所选 GPU 是否空闲；若忙，提示并重试/换卡，避免他人负载导致误判。
- 性能项单独计分，同时显示 TFLOPS / GB/s 与相对参考实现的百分比。

### 5.4 判分器自身校验（`make verify`）
- 将 `solutions/` 中的参考答案代入，必须全部 PASS（含性能项）。
- 将每个 `mutants/` 变体代入，必须 FAIL（每个 Assignment 至少 2–4 个变体，覆盖：差一错误、漏同步、缺边界检查、错误索引/layout）。
- 任何 Lab 完成的标准：`make verify` 全绿。

## 6. Tutorial（HTML）

- 每个 Lab 一个单页讲义，共用 `common/tutorial.css`；实现时借助 frontend-design skill 做视觉设计。
- 左侧目录导航；代码高亮；SVG 图示（线程层级、内存层级、layout 映射、tile distribution）。
- 关键概念配交互小部件（如 Lab 3 调 layout 参数实时显示"逻辑坐标 → 地址"映射）。
- 每节末尾标注对应 Assignment，"讲完即做"。
- 纯静态文件，浏览器本地打开；外部库仅从 CDN 按固定版本加载，同时支持浅色/深色模式。
- 内容中硬件参数（wavefront=64、LDS 64KB/CU、CU 数量等）以本机 `rocminfo` 实测为准。

## 7. 质量审查（每个 Lab 完成后）

派独立 subagent 并行审查：
1. **学生视角走查**：不看 solutions，只读 Tutorial + Assignment 尝试完成，报告不清楚之处、引导不足、知识跳跃。
2. **判分对抗审查**：编写"看似正确实则错误"的实现，检验测试是否能抓住；发现漏洞则补测试与 mutant。
3. **技术准确性审查**：核对 gfx942/CDNA3 硬件描述与 CK-Tile API 用法。

修复后复查，审查结论汇总到 `docs/reviews/labN.md`。

## 8. 环境与约束

- ROCm 7.2（`/opt/rocm`），8× MI308X（gfx942），CK-Tile 头文件位于 `/opt/rocm/include/ck_tile`，RCCL 已装；无 rocSHMEM、无 PyTorch。
- 机器共享：测试规模适中、运行有超时、自动选空闲卡。
- 依赖仅限 ROCm 自带工具 + Python 3 标准库。

## 9. CLAUDE.md

第一期一并建立，记录：课程定位与学习者背景、中文交流、语音识别对照（Hive=HIP，Cute/CkTile=CK-Tile）、目录与命名规范、挖空规范、判分规范（solutions + mutants + verify 全绿）、共享机器规则、Tutorial 风格约定、环境信息。

## 10. 不在第一期范围

Lab 4–8 的实现；多卡判分基础设施（第三期设计）。

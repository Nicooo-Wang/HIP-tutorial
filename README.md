# 从 HIP 到 Mega Kernel：MI308X 上的 GPU kernel 自学课程

一门在 AMD Instinct MI308X（gfx942 / CDNA3）上动手写 GPU kernel 的自学课程：从线程层级和访存讲起，经过 CK-Tile 与 MFMA tiled GEMM，再到多卡通信、通算融合，最后落到 mega kernel 技术。

**适合谁**：了解 GPU 编程的大致概念、但没系统写过 kernel；会用 C++、但模板元编程不熟；目标是在 MI308X 上做 mega kernel 方向的研究。

**怎么学**：每个 Lab 有一份讲义（`tutorial.html`，带图示和交互小部件）和几个作业（Assignment）。作业只用此前讲义讲过的概念；每个作业都能自动判分——正确性和性能都算分。

目标硬件（本机实测）：gfx942，80 CU，wavefront 64，每个 block（AMD 文档叫 workgroup）最多 1024 线程、LDS 64 KiB；ROCm 7.2（`/opt/rocm`）。

## 路线图

| Lab | 主题 | 作业 | 状态 |
|---|---|---|---|
| 0 [环境与工具](lab0-env/tutorial.html) | ROCm 工具链、hipcc 编译、错误检查、查询设备属性、rocprofv3 入门、判分流程 | a1 hello（第一个 kernel）· a2 device-query（查询设备属性） | 第一期 · 已完成 |
| 1 [HIP 基础](lab1-hip-basics/tutorial.html) | 线程层级、全局访存与合并、LDS、同步、并行归约、GEMM 分块与寄存器分块 | a1 vector-add（grid-stride loop）· a2 transpose（naive → LDS 分块）· a3 reduction（LDS 树形 → wavefront shuffle）· a4 gemm（naive → 自主设计 LDS 分块） | 第一期 · 已完成 |
| 2 [C++ 速成 + CK-Tile 入门](lab2-cktile-intro/tutorial.html) | `number` / `sequence` / `tuple` / `static_for`、constexpr 与编译期编程；tensor view → tile distribution → tile window → load/sweep/store_tile | a1 cpp-meta（6 个 C++ 小练习）· a2 ck-vector-add（第一个 CK-Tile kernel） | 第一期 · 已完成 |
| 3 [CK-Tile 核心概念](lab3-cktile-concepts/tutorial.html) | layout = 坐标变换的组合（tensor descriptor）、二维 tile distribution 与 LDS、pad 视图、在 tile 上归约 | a1 layout（tensor descriptor 与坐标变换）· a2 tile-transpose（CK-Tile LDS 转置）· a3 row-sum（自主设计行归约） | 第一期 · 已完成 |
| 4 MFMA 与 tiled GEMM | MFMA 指令、手写分块 GEMM、CK-Tile warp/block GEMM 组装 | 带性能门槛 | 第二期 |
| 5 GEMM 进阶 | 软件流水线、double buffer、epilogue 融合、Grouped GEMM | 带性能门槛 | 第二期 |
| 6 多卡通信基础 | P2P / IPC、RCCL、手写 all-to-all（device 端 flag 同步） | | 第三期 |
| 7 通算融合 | Grouped GEMM + all-to-all 融合（MoE 场景） | | 第三期 |
| 8 Mega kernel 技术 | persistent kernel、device 端任务队列与调度、跨 block 同步、指令级流水 | | 第四期 |

## 快速开始

```bash
cd lab0-env
make grade            # 编译并判分本 Lab 的全部作业（原样提交的骨架会有 FAIL，这是正常的）
```

讲义总目录：用浏览器直接打开 `site/index.html`（纯静态页面，`file://` 即可，不需要起服务器）。从 Lab 0 的讲义开始，按顺序往后学。

## 每个 Lab 的学习流程

1. **读讲义**：打开 `labN-*/tutorial.html`。每节末尾标注了“对应哪个作业的哪个 TODO”，读到那里就可以去做。
2. **做作业**：在 `aK-*/` 里编辑唯一的 `.hip` 文件，补全 `// TODO(aK-n): …（讲义 §x.y）` 标记、写在 `// >>> 你的代码 <<<` 区域里。自主设计题（见下）需要在标记之外新增 kernel 或辅助函数，位置写在该题的 README 和骨架注释里。先读该作业的 `README.md`：任务、用例表、分级 Hints（Hint 1 给方向，Hint 2 给关键步骤，个别作业还有折叠的 Hint 3），卡住时按顺序看。
3. **判分**：在 Lab 目录下运行
   ```bash
   make grade A=a2                     # 只判 a2；不加 A= 判整个 Lab
   make run A=a2 CASE=<用例名>         # 直接运行某个用例，打印详细信息（调试、printf、看性能数字）；
                                       # 用例名见 make grade 的输出或作业 README 的用例表，例如 Lab 1 的 make run A=a2 CASE=lds_bandwidth
   ```
4. **写书面问题**：答在 Lab 目录的 `answers.md` 里（不计分，但都是后面几课要用的直觉；很多题要你用 `make run` 做实验）。

难度随课程推进逐步过渡：先“填函数体”（签名和 launch 配置给定），再“补全设计”（自己定 block 尺寸、LDS 布局、循环结构），最后“自主设计”（只给接口和测试）。第一期各 Lab 的情况：Lab 0（环境）两题都是填空；Lab 1 从填函数体（a1、a3）、补全设计（a2 的 LDS 分块转置）走到自主设计（a4 的 LDS 分块 GEMM）；Lab 2 换了新框架，回到填空（a1 的 6 个 C++ 小函数、a2 的 4 个 TODO，常量和 grid 已给出）；Lab 3 从填空（a1）、补全设计（a2）走到自主设计（a3 行归约）。

## 判分是怎么做的

- 每个作业的 `test.cpp` 通过 `#include` 引入你的 `.hip` 文件；每个用例在独立进程里运行，单个用例超时 60 s，单次显存占用不超过 2 GB。
- **正确性用例**：多种尺寸（含 1、不整除、较大）、多个随机种子，容差按数据类型定；输出两侧有哨兵区（guard buffer），越界写会被发现；还会检查输入没有被修改。
- **性能用例**：同样计入分数。访存类作业以“同一进程里实测的 `hipMemcpyDeviceToDevice` 拷贝带宽”为基准，阈值是相对这个基准的比例，所以机器上有别人在跑时也相对公平；Lab 1 a4（GEMM）是绝对的 GFLOPS 阈值。性能用例失败会自动重试，最多跑 3 次取最好。阈值都经过校准：参考答案和“慢的错误实现”之间各留至少 1.5 倍余量，合理的简单设计能稳定通过。
- **“判分覆盖不到的 bug”**：多数作业的 README 有这一节，列出测试原理上抓不到的问题（数据竞争、未初始化变量、向量长度承诺过度等）。“全部 OK”不等于没有这些问题，请对照讲义自己排除。

## 关于 `solutions/`

`solutions/labN-*/aK-*/` 里是参考答案，以及用来校验判分器的错误实现（mutants）。**卡住超过 1 小时再看**——先把 README 的 Hint 按顺序看完、用 `make run` 定位到具体哪个用例、哪个位置出错。参考答案通常只是“一个简单设计”，不是最优解，性能用例的 README 里写了更好的设计能达到多少。

## 共享 GPU 说明

本机 8 张 MI308X 多人共享。`make grade` 会用 `rocm-smi` 自动挑最空闲的一张卡（输出第一行显示 `使用 GPU: HIP_VISIBLE_DEVICES=…`）；如果你已经设置了 `HIP_VISIBLE_DEVICES`，就尊重你的设置：

```bash
HIP_VISIBLE_DEVICES=3 make grade A=a2
```

`make run` 不自动选卡，需要时同样用 `HIP_VISIBLE_DEVICES` 指定。性能用例失败时先用 `rocm-smi` 看看那张卡是不是正忙，再重跑。

## 目录结构

```
HIP-tutorial/
├── README.md                 # 本文件
├── site/index.html           # 讲义总目录
├── common/                   # 公共基础设施：harness.hpp（测试框架）、grade.py（判分）、verify.py、讲义样式与脚本
├── labN-<name>/
│   ├── tutorial.html         # 讲义
│   ├── Makefile              # make grade / run / verify
│   ├── answers.md            # 书面问题作答
│   └── aK-<name>/{<name>.hip, test.cpp, README.md}   # 你只改 .hip
├── solutions/labN-<name>/aK-<name>/{<name>.hip, mutants/*.hip}
├── docs/                     # 设计文档、实施计划、审查记录
└── Makefile                  # 维护者用：make verify-all
```

## 给维护者

- `make verify-all`（仓库根目录）：先跑 `common/test_grade.py`（判分基础设施自测），再对每个 Lab 运行 `make verify`——参考答案必须满分、每个 mutant 至少挂一项、学生骨架能编译且不满分。全部 Lab 需要较长时间，并会占用 GPU。单个 Lab 用 `cd labN-* && make verify`。
- 审查记录在 `docs/reviews/`：每个 Lab 经过三份独立审查（学生视角走查、判分对抗、技术准确性，提示词见 `docs/reviews/PROMPTS.md`）；第一期的全量验证结果见 `docs/reviews/phase1-summary.md`。
- 项目约定见 `CLAUDE.md`：挖空格式、判分规范（阈值校准、mutant 必须确定性失败）、讲义风格，以及实测出来的 CK-Tile 已知坑。
- 新增一个 Lab：
  1. 建 `labN-<name>/`，`Makefile` 只写一行 `include ../common/lab.mk`；讲义以 `common/tutorial-template.html` 为起点，共用 `common/tutorial.css` / `tutorial.js`。
  2. 每个作业：`aK-<name>/{<name>.hip, test.cpp, README.md}`，`test.cpp` 只依赖 `common/harness.hpp`；参考答案和 ≥2 个确定性失败的 mutant 放在 `solutions/labN-<name>/aK-<name>/`（不要放进学生目录）。
  3. `make verify` 全绿后做三份审查，记录写入 `docs/reviews/labN.md`。
  4. 把 Lab 加进根目录 `Makefile` 的 `LABS`，并在 `site/index.html` 把对应卡片从“即将推出”改为链接，在本 README 的路线图里更新状态。

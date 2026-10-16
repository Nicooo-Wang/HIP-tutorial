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

（待补）

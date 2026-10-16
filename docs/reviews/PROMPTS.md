# 审查提示词（每个 Lab 完成后并行派发 3 个 subagent）

## R1 学生视角走查
你是一名会用 C++、了解 GPU 大致概念、但没写过 HIP kernel 的学生（CK-Tile 模板元编程不熟）。仓库根目录 /mnt/raid0/WangTianle/workspace/HIP-tutorial。
**禁止读取 solutions/ 目录与任何 mutants。** 只读 {LAB}/tutorial.html（可用浏览器或直接读 HTML）、{LAB}/aK-*/README.md 与骨架文件。
任务：按顺序在一个临时拷贝中完成每个 Assignment（`cp -r {LAB} /tmp/r1-{LAB}` 后在拷贝中做；`common/` 用原路径），用 `make grade` 检查。
报告（中文）：每个 Assignment 卡住的地方与原因（讲义缺了什么、Hint 不够、知识跳跃、TODO 说明有歧义）；讲义中引用但不存在的小节；判分输出是否帮你定位问题；最终得分。不要修改原仓库。

## R2 判分对抗审查
你的目标是找出判分漏洞：写出"看起来对但实际错误"的实现，看 `{LAB}` 的测试能否抓住。可以读 solutions/ 和 test.cpp。
对每个 Assignment 至少尝试 4 种不同错误（越界、漏边界、错误下标、数据竞争、只对特定尺寸成立、偷懒的性能作弊如不计算直接返回、写错误位置后又被覆盖等）。
用 `python3 common/grade.py {LAB} --only aK` 在临时目录的拷贝上测试（不要改原仓库文件）。
同时检查是否存在"正确实现被判错"的情况（例如合理的不同 block 尺寸、不同求和顺序、不同 tile 选择）——至少写 2 个与参考答案结构不同的正确实现并确认满分。
报告：漏网的错误实现（附代码）、被误判的正确实现（附代码）、建议新增的测试/mutant。

## R3 技术准确性审查
核对 {LAB}/tutorial.html、README 与代码注释中的技术事实：gfx942/CDNA3 硬件参数（CU 数、wave 大小、LDS 大小与 bank 数、缓存层级、峰值带宽/算力的计算）、HIP API 语义、CK-Tile API 用法（对照 /opt/rocm/include/ck_tile 源码）。可用 rocminfo、hipcc 写小程序实测。
报告：每条错误/不严谨陈述的位置、原文、更正及依据（源码路径或实测输出）。

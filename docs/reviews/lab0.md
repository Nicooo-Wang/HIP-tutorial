# Lab 0（lab0-env）审查记录

三份独立审查（提示词见 [PROMPTS.md](PROMPTS.md)，`{LAB}` = `lab0-env`），审查基线为提交 aa2066b 前后的 Lab 0。
修复分两批：判分侧（Task 10 batch A）与文档侧（Task 10 batch B）。

| 提交 | 内容 |
|---|---|
| 51a995a | harness：哨兵区每侧 256 KiB、填带载荷的 NaN 字；墙钟计时 |
| 9932fb1 | Lab 0 讲义、README、TODO 注释按 R1/R3 修订 |
| 962f440 | 公共样式：窄屏示意图最小宽度（与 Lab 1 文档一起提交） |

## R1 学生视角走查

结论：仅凭讲义与 README 一次通过（12/12），无阻塞。

| 发现 | 级别 | 处理 |
|---|---|---|
| `rocminfo` 不在默认 PATH；输出含 2 个 CPU agent（值为 0） | 重要 | 已修：a2 README 书面问题 2 与讲义 §2.3 表后说明改用 `/opt/rocm/bin/rocminfo` 并解释 CPU agent（9932fb1） |
| a2-6 隐式依赖 a2-1 的 `prop` | 重要 | 已修：TODO(a2-6) 写明“a2-1 中取得的 prop”；Hint 2 增加 prop 作用域提示（9932fb1） |
| a2-7 kHz→MHz 缺提示 | 重要 | 已修：Hint 2 给出两种写法的方向，TODO 注明 kHz ÷ 1000（9932fb1） |
| §1.4 编译命令与 lab.mk 不完全一致 | 小 | 已修：注明 lab.mk 用绝对路径、效果相同（9932fb1） |
| README rocprofv3 简写版不出 CSV | 小 | 已修：README 写出完整命令与 `prof/a1_kernel_trace.csv`（9932fb1） |
| block=2048 的报错文字与枚举名对不上 | 小 | 已修：§4.2 与 README 注明 `invalid configuration argument` = `hipErrorInvalidConfiguration`（9932fb1） |
| a1-1 骨架提示直接给出全局下标公式 | 小 | 已修：骨架改为指向 §3.2 的类比，不再重复公式（9932fb1） |
| （控制者裁定）§3.3 与 a1 Hint 2 直接给出向上取整写法 | — | 已修：§3.3 改为给出判断标准、让学生自己推出常数并验算；Hint 2 改为引导问题（9932fb1） |

## R2 判分对抗审查

结论：漏网 10 个错误实现，误判正确实现 0 个。

| 发现 | 处理 |
|---|---|
| S1 伪造 `@@RESULT` 后 `_exit`、`#define` 改写测试宏 | 搁置：属于有意作弊。本课程是自学课程，作弊只骗自己；不做源码扫描 |
| S2 只测 device 0、硬编码本机数值 | 搁置：本机 8 卡同型号，判分只看得见一张卡，无法区分。a2 README 新增说明“测试分辨不出写死的数字，请按表用 API 查询”（9932fb1） |
| S3 错误属性但数值巧合（warp = maxThreads/16、arch 固定前 6 字节、四舍五入频率） | 搁置：同 S2，本机数值巧合无法区分；README 的同一说明覆盖 |
| S4 稀疏越界写跳过 4 KiB 哨兵（w01 `out[n+4000]`、w11） | 已修：哨兵区扩大到每侧 256 KiB（51a995a），w01、w11 现在 0/6；负向偏移 w08 每个用例都被抓到 |
| S5 不写 kernel（主机循环）、忽略 stream | 搁置：单 stream 测试下不可观测；讲到 stream 的 Lab 再考虑 |

## R3 技术准确性审查

结论：错误 2、不严谨 6、建议 7，全部处理（9932fb1）。

| 发现 | 处理 |
|---|---|
| E1 a1 README 的 `rocprofv3 --kernel-trace` 不生成 CSV | 已修：改为完整命令，并提示按 `Grid_Size_X` 认出 many_blocks |
| E2 “hipDeviceGetAttribute 开销更小” | 已修：§5.2 改为“写起来更直接”，并说明 ROCm 实现每次都构造整份 prop、两者都 < 1 µs |
| U1 非法地址时“总会打印 Memory access fault” | 已修：§4.2 改为“部分情况下才打印 HSA_STATUS_ERROR_…，常见越界只有错误码”；pitfall 补充“写空指针附近也可能返回 hipSuccess” |
| U2 图 2.1“LDS → L1 → … 越来越大” | 已修：图注改为缓存层级 L1→L2→IC→HBM，LDS 与 L1 同级、不是缓存 |
| U3 “每个成员都有注释说明单位” | 已修：§5.1“多数成员有注释、部分注明单位”；§5.2 tip 补“没写单位时用已知值反推” |
| U4 hipDeviceSynchronize 等待范围 | 已修：“本进程、当前设备、所有 stream” |
| U5 hipMemcpy 等 kernel 的前提 | 已修：图 4.1 图注注明“同一（默认）stream” |
| U6 strncpy 不补 '\0' 的条件 | 已修：`strlen(src) >= len` |
| S1 rocminfo 路径 | 已修（同 R1） |
| S2 amd-smi “MB” 实为 MiB | 已修：a2 README 探索任务注明 |
| S3 L2 工具只报一个实例 | 已修：§2.3 表后说明 |
| S4 SPX 分区前提 | 已修：§2.3 表后说明 |
| S5 wave64 物理上 16 lane × 4 周期 | 已修：§2.2 括注 |
| S6 为什么必须 `--output-format csv` | 已修：§6.3 补一句 |
| S7 MaxRegistersPerBlock 的来由 | 未改：审查者标为可选，正文数值正确 |

另外按 Task 10 batch A 的判分变化同步了文档：哨兵区描述（§1.3 文字与图 1.2：256 KiB、NaN 字，越界读参与运算会污染结果）、a1 README 的哨兵大小。

## 校准

Lab 0 没有性能用例。

## 最终 `make verify`

```
== a1-hello
   solution    满分 6/6
   mutant      m1_no_bounds.hip                 被抓住: single_element, not_multiple_of_block, many_blocks
   mutant      m2_thread_only.hip               被抓住: not_multiple_of_block, many_blocks
   skeleton    0/6
== a2-device-query
   solution    满分 6/6
   mutant      m1_hardcoded_cu.hip              被抓住: compute_units_and_wave
   mutant      m2_lds_wrong_attr.hip            被抓住: block_limits
   skeleton    0/6

VERIFY OK
```

骨架原样 `make grade A=a1` 的输出与讲义 §1.3 示例逐行一致（0/6，“得到 -1”）。

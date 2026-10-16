# Lab 0 · a2 device-query：查询设备属性

## 你要做什么

编辑 `a2-device-query/device_query.hip`，补全

```cpp
DeviceSummary query_device(int device);
```

用 HIP API 查询第 `device` 号 GPU，把结构体 `DeviceSummary` 的 7 个字段填成**本机实测值**。结构体每个字段的注释写明了该用哪个属性；每个字段对应一个 TODO（`a2-1` 至 `a2-7`）：

| 字段 | 来源 | 讲义 |
|------|------|------|
| `arch` | `hipDeviceProp_t::gcnArchName` 中第一个 `:` 之前的部分 | §5.3 |
| `num_cu` | `hipDeviceAttributeMultiprocessorCount` | §5.2 |
| `warp_size` | `hipDeviceAttributeWarpSize` | §5.2 |
| `max_threads_per_block` | `hipDeviceAttributeMaxThreadsPerBlock` | §5.2 |
| `lds_bytes_per_block` | `hipDeviceAttributeMaxSharedMemoryPerBlock` | §5.2 |
| `global_mem_bytes` | `hipDeviceProp_t::totalGlobalMem`（字节） | §5.1 |
| `clock_mhz` | `hipDeviceAttributeClockRate`（kHz）换算成 MHz | §5.2 |

不允许写死任何数字——同一份代码换到另一张卡上也必须正确。每个 HIP 调用都用 `HIP_CHECK(...)` 包起来（讲义 §4.3）。

> **测试分辨不出“写死的数字”。** 本机 8 张卡完全相同，判分时也只看得见一张卡，所以把本机的值手写进去（或者用“碰巧数值相等”的别的属性凑出来）照样能拿满分——但那样这道题就白做了，换一台机器（甚至换一种分区模式）就会错。请按上表用 API 查询，这是这道题真正要练的东西。

## 怎么判分

在 `lab0-env/` 目录下运行 `make grade A=a2`。测试在同一进程里用 HIP API 独立查一遍“标准答案”，再和你返回的结构体逐字段比较；失败信息会给出字段名、期望值和你的值。

| 用例 | 分值 | 检查什么 |
|------|------|----------|
| `arch_name` | 1 | `arch` 与 `gcnArchName` 冒号前的部分完全相同（且以 `'\0'` 结尾） |
| `compute_units_and_wave` | 2 | `num_cu`、`warp_size` |
| `block_limits` | 2 | `max_threads_per_block`、`lds_bytes_per_block` |
| `memory_and_clock` | 1 | `global_mem_bytes`、`clock_mhz` |

测试总是查询 `device 0`——判分脚本用 `HIP_VISIBLE_DEVICES` 选中的那张卡在程序里就是 0 号（讲义 §6.2）。

## Hints

<details><summary>Hint 1（方向）</summary>

先调用一次 `hipGetDeviceProperties` 拿到整个 `hipDeviceProp_t`，`arch` 和 `global_mem_bytes` 从它里面取；其余 5 个字段各用一次 `hipDeviceGetAttribute`。注意每个属性的单位，以及 `totalGlobalMem` 的类型。

</details>

<details><summary>Hint 2（关键步骤）</summary>

- `hipDeviceGetAttribute` 的第一个参数是 `int*`：直接传结构体字段的地址即可；第三个参数是 `device`（用函数参数，别写 0）。
- `prop`：a2-1 里取得的 `hipDeviceProp_t` 变量在 a2-6 还要用（读 `totalGlobalMem`），所以把它声明在函数开头、两处都能看到的地方，只查一次即可。
- 频率：属性返回 kHz，要换成 MHz。可以先查进一个 `int` 临时变量再除以 1000 赋给 `s.clock_mhz`；也可以直接查进 `s.clock_mhz`，再对它原地除以 1000（整数除法即可）。
- `arch`：`std::strchr` 找冒号 → 指针相减得长度 → 长度不超过 `sizeof(s.arch) - 1` → 复制 → 在末尾写 `'\0'`（讲义 §5.3 的三步）。

</details>

## 书面问题（写到 ../answers.md）

1. 本机 GPU 有多少个 CU？
2. 每个 CU 理论上最多能同时驻留多少个 wavefront？（已知：每个 SIMD 最多驻留 8 个 wave——本机 KFD 拓扑 `/sys/class/kfd/kfd/topology/nodes/*/properties` 中 `max_waves_per_simd 8`；每个 CU 有 4 个 SIMD，见讲义 §2.1。算完后可用 `/opt/rocm/bin/rocminfo | grep -iE "SIMDs per CU|Max Waves Per CU"` 验证——`rocminfo` 不在默认 `PATH` 里，要写全路径；输出里前两行来自 CPU agent（值为 0），看后面 GPU agent 的那些行。）整张 GPU 最多同时驻留多少个 wavefront、多少个线程？
3. `max_threads_per_block = 1024` 的一个 block 由几个 wavefront 组成？blockDim = 100 时呢（最后一个 wavefront 有多少个 lane 是空闲的）？

## 动手探索（不判分）

- 在 `query_device` 末尾临时加一行 `printf` 把结构体打印出来，用 `make run A=a2` 运行；再换一张卡：`HIP_VISIBLE_DEVICES=3 make run A=a2`。8 张卡的值一样吗？
- `global_mem_bytes` 换算成 GiB（除以 2³⁰）是多少？和 `amd-smi static -g 0` 中 `VRAM` 一节的 `SIZE` 对照（注意：amd-smi 标的 “MB” 实际是 MiB，即 2²⁰ 字节）。

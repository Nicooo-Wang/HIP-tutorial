// Lab 0 · a2 device-query：用 HIP API 查询本机 GPU 的关键参数
// 任务：补全 query_device(device)，把下列 7 个字段填成"在本机实测"的值——不要手写常数。
// 判分：在 lab0-env/ 目录下运行 make grade A=a2；调试：make run A=a2 CASE=用例名
// HIP_CHECK 宏由 common/harness.hpp 提供（test.cpp 在包含本文件之前已包含它），用法见讲义 §4.3。
#include <hip/hip_runtime.h>
#include <cstring>

struct DeviceSummary {
  char arch[64];                  // gcnArchName 中冒号前部分，如 "gfx942"
  int  num_cu;                    // hipDeviceAttributeMultiprocessorCount
  int  warp_size;                 // hipDeviceAttributeWarpSize
  int  max_threads_per_block;     // hipDeviceAttributeMaxThreadsPerBlock
  int  lds_bytes_per_block;       // hipDeviceAttributeMaxSharedMemoryPerBlock
  long long global_mem_bytes;     // hipDeviceProp_t::totalGlobalMem
  int  clock_mhz;                 // hipDeviceAttributeClockRate / 1000
};

DeviceSummary query_device(int device) {
  DeviceSummary s{};  // 所有字段先置 0、arch 先置空串

  // TODO(a2-1): arch —— 先用 hipGetDeviceProperties 取得 hipDeviceProp_t，再把 gcnArchName
  //   （形如 "gfx942:sramecc+:xnack-"）中第一个 ':' 之前的部分复制进 s.arch，别忘了 '\0' 结尾（讲义 §5.3）
  // >>> 你的代码 <<<

  // TODO(a2-2): num_cu —— 查询属性 hipDeviceAttributeMultiprocessorCount（讲义 §5.2）
  // >>> 你的代码 <<<

  // TODO(a2-3): warp_size —— 查询属性 hipDeviceAttributeWarpSize（讲义 §5.2）
  // >>> 你的代码 <<<

  // TODO(a2-4): max_threads_per_block —— 查询属性 hipDeviceAttributeMaxThreadsPerBlock（讲义 §5.2）
  // >>> 你的代码 <<<

  // TODO(a2-5): lds_bytes_per_block —— 查询属性 hipDeviceAttributeMaxSharedMemoryPerBlock（讲义 §5.2）
  // >>> 你的代码 <<<

  // TODO(a2-6): global_mem_bytes —— 读 a2-1 中取得的 hipDeviceProp_t（prop）的成员 totalGlobalMem（单位：字节）（讲义 §5.1）
  // >>> 你的代码 <<<

  // TODO(a2-7): clock_mhz —— 查询属性 hipDeviceAttributeClockRate（单位 kHz），再换算成 MHz：kHz ÷ 1000（讲义 §5.2）
  // >>> 你的代码 <<<

  return s;
}

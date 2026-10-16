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
  DeviceSummary s{};
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  const char* colon = std::strchr(prop.gcnArchName, ':');
  size_t len = colon ? size_t(colon - prop.gcnArchName) : std::strlen(prop.gcnArchName);
  if (len > sizeof(s.arch) - 1) len = sizeof(s.arch) - 1;
  std::strncpy(s.arch, prop.gcnArchName, len);
  s.arch[len] = '\0';

  s.num_cu = 304;  // mutant: 硬编码 MI300X 的 CU 数，而不是查询本机
  HIP_CHECK(hipDeviceGetAttribute(&s.warp_size, hipDeviceAttributeWarpSize, device));
  HIP_CHECK(hipDeviceGetAttribute(&s.max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, device));
  HIP_CHECK(hipDeviceGetAttribute(&s.lds_bytes_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
  s.global_mem_bytes = (long long)prop.totalGlobalMem;
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, device));
  s.clock_mhz = khz / 1000;
  return s;
}

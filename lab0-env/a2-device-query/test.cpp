#include "harness.hpp"
#include STUDENT_SRC
#include <string>

static DeviceSummary expected(int dev) {
  DeviceSummary e{};
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, dev));
  std::string a = p.gcnArchName;
  a = a.substr(0, a.find(':'));
  std::snprintf(e.arch, sizeof(e.arch), "%s", a.c_str());
  HIP_CHECK(hipDeviceGetAttribute(&e.num_cu, hipDeviceAttributeMultiprocessorCount, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.warp_size, hipDeviceAttributeWarpSize, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.lds_bytes_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
  e.global_mem_bytes = (long long)p.totalGlobalMem;
  int khz;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, dev));
  e.clock_mhz = khz / 1000;
  return e;
}

#define EXPECT_FIELD(f) if (g.f != e.f) hk::fail(R, #f ": 期望 %lld，得到 %lld", (long long)e.f, (long long)g.f)

TEST_CASE(arch_name, 1) {
  auto g = query_device(0), e = expected(0);
  if (std::string(g.arch) != e.arch) hk::fail(R, "arch: 期望 \"%s\"，得到 \"%s\"", e.arch, g.arch);
}
TEST_CASE(compute_units_and_wave, 2) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(num_cu); EXPECT_FIELD(warp_size); }
TEST_CASE(block_limits, 2) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(max_threads_per_block); EXPECT_FIELD(lds_bytes_per_block); }
TEST_CASE(memory_and_clock, 1) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(global_mem_bytes); EXPECT_FIELD(clock_mhz); }
HK_MAIN()

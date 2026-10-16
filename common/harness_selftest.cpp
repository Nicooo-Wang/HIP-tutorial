#include "harness.hpp"

__global__ void write_n_plus_one(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) p[i] = 1.0f;  // 故意越界写一个元素
}
__global__ void write_n(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f;
}

TEST_CASE(guard_detects_overflow, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_n_plus_one<<<4, 256>>>(b.get(), 1000);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "越界写未被检测到");
}

TEST_CASE(guard_accepts_inbounds, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_n<<<4, 256>>>(b.get(), 1000);
  HIP_CHECK(hipDeviceSynchronize());
  b.guards_intact(R);
  auto h = b.download();
  for (float v : h) if (v != 1.0f) { hk::fail(R, "数据区错误"); break; }
}

TEST_CASE(guard_patterns_differ, 1) {
  hk::GuardedBuffer<float> a(16), b(16);
  if (a.pattern == b.pattern) hk::fail(R, "相邻两个缓冲区的哨兵模式相同");
}

TEST_CASE(check_close_behaviour, 1) {
  hk::Result a, b, c;
  std::vector<double> ref = {1.0, 2.0, 3.0};
  if (!hk::check_close(a, std::vector<float>{1.0f, 2.0f, 3.0f}, ref, 1e-6, 1e-6, "ok")) hk::fail(R, "相等误判");
  if (hk::check_close(b, std::vector<float>{1.0f, 2.5f, 3.0f}, ref, 1e-6, 1e-6, "bad")) hk::fail(R, "不等未判出");
  if (b.msg.find("[1]") == std::string::npos) hk::fail(R, "失败信息未包含下标: %s", b.msg.c_str());
  if (hk::check_close(c, std::vector<float>{1.0f, NAN, 3.0f}, ref, 1e9, 1e9, "nan")) hk::fail(R, "NaN 未判出");
}

TEST_CASE(always_fails_for_protocol_check, 0) {
  hk::fail(R, "预期失败: 含 \"引号\" 与\n换行");
}

PERF_CASE(copy_bandwidth_positive, 0) {
  R.metric = hk::copy_bandwidth_gbs(size_t(256) << 20);
  R.unit = "GB/s";
  if (!(R.metric > 100.0)) hk::fail(R, "拷贝带宽异常: %.1f", R.metric);
}

HK_MAIN()

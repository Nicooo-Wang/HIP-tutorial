#include "harness.hpp"
#include <memory>

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

__global__ void write_minus_one(float* p) { p[-1] = 1.0f; }

TEST_CASE(guard_detects_underflow, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_minus_one<<<1, 1>>>(b.get());
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "前向越界写未被检测到");
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

TEST_CASE(guard_patterns_cross_type, 1) {
  hk::GuardedBuffer<float> a(16);
  hk::GuardedBuffer<int> b(16);
  if (a.pattern == b.pattern) hk::fail(R, "float/int 缓冲区哨兵相同");
  std::vector<std::unique_ptr<hk::GuardedBuffer<float>>> v;
  for (int i = 0; i < 7; ++i) v.emplace_back(new hk::GuardedBuffer<float>(8));
  for (int i = 1; i < 7; ++i)
    if (v[i]->pattern == v[i - 1]->pattern) hk::fail(R, "第 %d 个缓冲区与前一个哨兵相同", i);
}

TEST_CASE(check_equal_behaviour, 1) {
  hk::Result a, b, c;
  if (!hk::check_equal(a, std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 3}, "ok")) hk::fail(R, "相等误判");
  if (hk::check_equal(b, std::vector<int>{1, 9, 3}, std::vector<int>{1, 2, 3}, "bad")) hk::fail(R, "不等未判出");
  if (b.msg.find("[1]") == std::string::npos) hk::fail(R, "失败信息未包含下标: %s", b.msg.c_str());
  if (hk::check_equal(c, std::vector<int>{1}, std::vector<int>{1, 2}, "len")) hk::fail(R, "长度不符未判出");
}

TEST_CASE(check_close_size_and_inf, 1) {
  hk::Result a, b, c;
  if (hk::check_close(a, std::vector<float>{1.f}, std::vector<double>{1.0, 2.0}, 1e-6, 1e-6, "len"))
    hk::fail(R, "长度不符未判出");
  const float inf = INFINITY;
  if (!hk::check_close(b, std::vector<float>{inf}, std::vector<double>{double(inf)}, 1e-6, 1e-6, "inf"))
    hk::fail(R, "Inf 精确匹配被拒绝");
  if (hk::check_close(c, std::vector<float>{1.f}, std::vector<double>{double(inf)}, 1e9, 1e9, "inf2"))
    hk::fail(R, "有限值匹配 Inf 参考未被拒绝");
}

TEST_CASE(always_fails_for_protocol_check, 0) {
  hk::fail(R, "预期失败: 含 \"引号\" 与\n换行 与\t制表符");
}

PERF_CASE(copy_bandwidth_positive, 0) {
  R.metric = hk::copy_bandwidth_gbs(size_t(256) << 20);
  R.unit = "GB/s";
  if (!(R.metric > 100.0)) hk::fail(R, "拷贝带宽异常: %.1f", R.metric);
}

TEST_CASE(json_escape_control_char, 1) {
  if (hk::json_escape("a\tb") != "a\\u0009b") hk::fail(R, "控制字符未转义为 \\u0009");
}

HK_MAIN()

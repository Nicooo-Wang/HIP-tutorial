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

__global__ void write_canonical_nan(float* p) { p[0] = __int_as_float(0x7FC00000); }
__global__ void write_u16(unsigned short* p, int i) { p[i] = 0; }

__global__ void write_at(float* p, long long off) { p[off] = 1.0f; }

TEST_CASE(guard_detects_far_overflow, 1) {
  // 跳过数据区末尾约 100 KiB 的稀疏越界写也要能发现
  hk::GuardedBuffer<float> b(1000);
  write_at<<<1, 1>>>(b.get(), 1000 + 25000);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "距末尾约 100 KiB 的越界写未被检测到");
  else if (tmp.msg.find("之后") == std::string::npos) hk::fail(R, "方向报告错误: %s", tmp.msg.c_str());
}

TEST_CASE(guard_detects_far_underflow, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_at<<<1, 1>>>(b.get(), -25000);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "距开头约 100 KiB 的前向越界写未被检测到");
  else if (tmp.msg.find("之前") == std::string::npos || tmp.msg.find("99997 字节") == std::string::npos)
    hk::fail(R, "方向/距离报告错误: %s", tmp.msg.c_str());
}

TEST_CASE(guard_detects_canonical_nan_write, 1) {
  // 往哨兵区写入规范 NaN（GPU 运算产生的 NaN）也必须被发现：哨兵的 NaN 载荷非 0
  hk::GuardedBuffer<float> b(1000);
  write_canonical_nan<<<1, 1>>>(b.get() + 1000);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "写入规范 NaN 的越界写未被检测到");
}

// 越界读到的值哪怕乘以 0 也必须污染结果（哨兵是 NaN）
__global__ void oob_read_times_zero(const float* in, float* out, int n) {
  out[0] = in[0] + 0.0f * in[n];       // 紧邻末尾
  out[1] = in[0] + 0.0f * in[n + 4096];  // 末尾之后 16 KiB
  out[2] = in[0] + 0.0f * in[-1];      // 紧邻开头
  out[3] = in[0] + 0.0f * in[-30000];  // 开头之前约 120 KiB
}

TEST_CASE(guard_oob_read_times_zero_is_nan, 1) {
  hk::GuardedBuffer<float> in(1000), out(4);
  in.fill(2.0f);
  oob_read_times_zero<<<1, 1>>>(in.get(), out.get(), 1000);
  auto o = out.download();
  for (int k = 0; k < 4; ++k)
    if (!std::isnan(o[k])) hk::fail(R, "第 %d 处越界读 ×0 没有得到 NaN（得到 %g）", k, o[k]);
  out.guards_intact(R);
  in.guards_intact(R);
}

TEST_CASE(guard_pattern_is_payload_nan, 1) {
  hk::GuardedBuffer<float> a(3), b(5);
  for (auto* g : {&a, &b}) {
    float f;
    std::memcpy(&f, &g->pattern, 4);
    if (!std::isnan(f) || (g->pattern & 0x7FC00000u) != 0x7FC00000u || (g->pattern & 0x3FFFFFu) == 0)
      hk::fail(R, "哨兵 0x%08x 不是带非零载荷的 quiet NaN", g->pattern);
  }
  // 数据区初值也是本缓冲区的哨兵（未写过的输出读出来是 NaN）
  auto h = a.download();
  for (float v : h) if (!std::isnan(v)) { hk::fail(R, "数据区初值不是 NaN"); break; }
}

TEST_CASE(guard_unaligned_tail, 1) {
  // 2 字节元素、奇数个：数据区末尾不在 4 字节边界上，紧随其后的 2 个字节也属于哨兵
  hk::GuardedBuffer<unsigned short> b(7), c(7);
  hk::Result ok;
  if (!b.guards_intact(ok)) hk::fail(R, "未越界却报告越界: %s", ok.msg.c_str());
  write_u16<<<1, 1>>>(c.get(), 7);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (c.guards_intact(tmp)) hk::fail(R, "2 字节元素紧邻末尾的越界写未被检测到");
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

// FMA 微基准（计算类性能用例的基准）：空闲 MI308X 约 27 TFLOPS，同卡被别的 GEMM 持续占满时也有约 11 TFLOPS。
// 下限只用来抓"kernel 没跑 / 被编译器删掉 / 计时出错"这类问题。
PERF_CASE(fma_peak_positive, 0) {
  R.metric = hk::fma_peak_gflops();
  R.unit = "GFLOPS";
  if (!(R.metric > 2000.0 && R.metric < 200000.0)) hk::fail(R, "FMA 吞吐异常: %.1f GFLOPS", R.metric);
}

TEST_CASE(json_escape_control_char, 1) {
  if (hk::json_escape("a\tb") != "a\\u0009b") hk::fail(R, "控制字符未转义为 \\u0009");
}

HK_MAIN()

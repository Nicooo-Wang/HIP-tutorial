#include "harness.hpp"
#include STUDENT_SRC

using Fn = void (*)(const float*, int, float*, float*, hipStream_t);

// 返回 GPU 结果；out 与 ws 预先填 NaN，以抓住"未初始化就累加"
static float run(Fn fn, const std::vector<float>& h) {
  int n = int(h.size());
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.upload(h); out.fill(NAN); ws.fill(NAN);
  fn(in.get(), n, out.get(), ws.get(), nullptr);
  hk::Result tmp;
  if (!out.guards_intact(tmp) || !ws.guards_intact(tmp)) return INFINITY;  // 越界写 → 判错
  if (in.download() != h) return INFINITY;
  return out.download()[0];
}

// 全 1 输入：n < 2^24 时任何求和顺序都精确等于 n，可做精确比较，抓住"漏掉/重复元素"
static void ones(hk::Result& R, Fn fn, const char* name) {
  for (int n : {1, 2, 63, 64, 65, 255, 256, 257, 1000, 4097, (1 << 20) + 1, 3 * (1 << 20) + 5}) {
    float g = run(fn, std::vector<float>(n, 1.0f));
    if (g != float(n)) { hk::fail(R, "%s: 全 1 输入 n=%d，期望 %d，得到 %.9g", name, n, n, g); return; }
  }
}

static void randoms(hk::Result& R, Fn fn, const char* name) {
  for (int n : {1000, 1 << 20, (1 << 24) + 3}) {
    auto h = hk::random_floats(n, n);
    double ref = 0, mag = 0;
    for (float x : h) { ref += x; mag += std::fabs(x); }
    float g = run(fn, h);
    double tol = 1e-5 * mag + 1e-6;
    if (!(std::fabs(g - ref) <= tol)) { hk::fail(R, "%s: 随机输入 n=%d，期望 %.9g，得到 %.9g（容差 %.3g）", name, n, ref, g, tol); return; }
  }
}

TEST_CASE(lds_exact_ones, 3)   { ones(R, reduce_sum_lds, "lds"); }
TEST_CASE(lds_random, 2)       { randoms(R, reduce_sum_lds, "lds"); }
TEST_CASE(shfl_exact_ones, 3)  { ones(R, reduce_sum_shfl, "shfl"); }
TEST_CASE(shfl_random, 2)      { randoms(R, reduce_sum_shfl, "shfl"); }
TEST_CASE(repeatable_output, 1) {
  // 连续调用两次必须得到同样的结果（第二次调用时 out 中已有上次的值）
  int n = 1 << 20;
  auto h = hk::random_floats(n, 99);
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.upload(h); out.fill(NAN); ws.fill(NAN);
  reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr);
  float a = out.download()[0];
  reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr);
  float b = out.download()[0];
  double ref = 0, mag = 0;
  for (float x : h) { ref += x; mag += std::fabs(x); }
  if (!(std::fabs(a - ref) <= 1e-5 * mag + 1e-6)) { hk::fail(R, "第一次调用结果错误：期望 %.9g，得到 %.9g", ref, a); return; }
  if (!(std::fabs(a - b) <= 1e-5 * n)) hk::fail(R, "两次调用结果不同：%.9g vs %.9g（是否把结果累加到了旧的 out 上？）", a, b);
}

constexpr double kReduceRatio = 0.45;  // 校准：参考答案约 0.75×拷贝带宽，单 block 版约 0.002×（S/N>10）；取 floor2(S/1.67)，对参考答案留 1.5× 余量
PERF_CASE(shfl_bandwidth, 3) {
  int n = 1 << 26;
  size_t bytes = size_t(n) * sizeof(float);
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.fill(1.0f);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr); });
  R.metric = bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  float g = out.download()[0];
  if (!(std::fabs(g - double(n)) <= 1e-6 * n)) { hk::fail(R, "结果错误：期望 %d，得到 %.9g", n, g); return; }
  if (R.metric < kReduceRatio * copy)
    hk::fail(R, "读带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kReduceRatio * copy, kReduceRatio * 100, copy);
}
HK_MAIN()

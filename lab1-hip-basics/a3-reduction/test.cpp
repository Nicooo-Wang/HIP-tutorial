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

// 全 1 输入：n < 2^24 时任何求和顺序都精确等于 n，可做精确比较，抓住"漏掉/重复元素"。
// 含 n ≥ 2^22 且 n % 4 ≠ 0 的大尺寸（抓住只在大 n 走向量化路径时丢掉尾部的写法）；
// 整组尺寸重复 3 遍，提高偶发数据竞争被撞上的概率（竞争类错误不保证每次都能抓到）。
static void ones(hk::Result& R, Fn fn, const char* name) {
  for (int rep = 0; rep < 3; ++rep)
    for (int n : {1, 2, 63, 64, 65, 255, 256, 257, 1000, 4097, (1 << 20) + 1, 3 * (1 << 20) + 5,
                  (1 << 22) + 3, (1 << 23) + 1, (1 << 24) - 1}) {
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

// 校准（墙钟计时，gfx942 空闲卡）：参考答案 0.656–0.676×拷贝带宽（最差 S_min=0.656），单 block 版（m3）约 0.0015×。
// 合理的正确设计也必须稳定通过：第 1 遍只用 256 个 block 的参考结构（kMaxBlocks=256）约 0.40×。
// 因此取 0.28：参考答案最差一次 2.3× 余量，256-block 变体约 1.43×，单 block 版低于阈值约 190×。
// （原为 0.45；改用 hipDeviceSynchronize 夹住的墙钟计时后参考答案比值从约 0.70 降到约 0.67。）
constexpr double kReduceRatio = 0.28;
// 输入是随机的 0/1（约 1/8 为 1，首尾若干个强制为 1）：所有元素非负，任何求和顺序下的每个部分和都 ≤ 总和 < 2^24，
// 都是可精确表示的整数，所以结果必须与 CPU 精确相等——不会误判任何求和顺序的正确实现，又能抓住漏掉/重复元素。
PERF_CASE(shfl_bandwidth, 3) {
  const int n = (1 << 26) - 1;
  size_t bytes = size_t(n) * sizeof(float);
  std::vector<float> h(n);
  hk::Rng rng(2026);
  long long ref = 0;
  for (int i = 0; i < n; ++i) {
    h[i] = (i < 4 || i >= n - 4 || (rng.next() & 7) == 0) ? 1.0f : 0.0f;
    ref += h[i] != 0.0f;
  }
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.upload(h); out.fill(NAN); ws.fill(NAN);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr); });
  R.metric = bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  if (!out.guards_intact(R) || !ws.guards_intact(R)) return;
  float g = out.download()[0];
  if (double(g) != double(ref)) { hk::fail(R, "结果错误：n=%d 的随机 0/1 输入，期望 %lld，得到 %.9g", n, ref, g); return; }
  if (in.download() != h) { hk::fail(R, "输入被修改了"); return; }
  if (R.metric < kReduceRatio * copy)
    hk::fail(R, "读带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kReduceRatio * copy, kReduceRatio * 100, copy);
}
HK_MAIN()

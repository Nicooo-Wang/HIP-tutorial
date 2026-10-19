#include "harness.hpp"
#include STUDENT_SRC

using Fn = void (*)(const float*, const float*, float*, int, int, int, hipStream_t);

static void run(hk::Result& R, Fn fn, const char* name, int M, int N, int K, uint64_t seed) {
  auto ha = hk::random_floats(size_t(M) * K, seed), hb = hk::random_floats(size_t(K) * N, seed + 1);
  hk::GuardedBuffer<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
  A.upload(ha); B.upload(hb); C.fill(NAN);
  fn(A.get(), B.get(), C.get(), M, N, K, nullptr);
  auto hc = C.download();
  // 逐元素误差界：|err| ≤ 1e-5 · Σ_k |a_ik·b_kj| + 1e-6（比随机输入下 fp32 的典型舍入误差 ~√K·u·Σ|ab| 宽松得多，对任何合理的求和顺序都成立；它不是最坏情况的严格上界）
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double ref = 0, mag = 0;
      for (int k = 0; k < K; ++k) {
        double p = double(ha[size_t(i) * K + k]) * hb[size_t(k) * N + j];
        ref += p; mag += std::fabs(p);
      }
      double g = hc[size_t(i) * N + j];
      if (!(std::fabs(g - ref) <= 1e-5 * mag + 1e-6)) {
        hk::fail(R, "%s M=%d N=%d K=%d: C[%d][%d] 期望 %.7g，得到 %.7g", name, M, N, K, i, j, ref, g);
        return;
      }
    }
  if (!C.guards_intact(R)) return;
  if (A.download() != ha || B.download() != hb) hk::fail(R, "%s: 输入被修改了", name);
}

static const int kShapes[][3] = {{1, 1, 1}, {40, 70, 33}, {17, 33, 65}, {64, 64, 64}, {128, 96, 200}, {500, 300, 257}, {512, 512, 512}};

TEST_CASE(naive_small, 1) { run(R, launch_gemm_naive, "naive", 4, 5, 6, 1); }
TEST_CASE(naive_all_shapes, 3) {
  for (auto& s : kShapes) { run(R, launch_gemm_naive, "naive", s[0], s[1], s[2], 2); if (!R.passed) return; }
}
TEST_CASE(tiled_aligned, 2) { run(R, launch_gemm_tiled, "tiled", 128, 128, 128, 3); }
TEST_CASE(tiled_all_shapes, 4) {
  for (auto& s : kShapes) { run(R, launch_gemm_tiled, "tiled", s[0], s[1], s[2], 4); if (!R.passed) return; }
}

// 阈值 = kTiledRatio × 同进程实测 fp32 FMA 吞吐（hk::fma_peak_gflops，纯寄存器 FMA 微基准）。
// 学生 kernel 与微基准交替计时、各取最优：别人在同一张卡上跑东西时两者一起变慢，比值基本不变
// （绝对阈值 2000 GFLOPS 在 3 个并发参考答案进程下会误判正确代码：参考答案从约 4630 掉到约 2440）。
// 校准（墙钟，空闲 GPU 2、5 各 3 次；FMA 吞吐约 26.8 TFLOPS）：参考答案（32×32 tile、32×8 block、每线程 4 输出）
// 0.1728–0.1730（约 4630 GFLOPS）；16×16 tile、每线程 1 输出的合理变体 0.1150–0.1152（约 3085）；naive（m3）0.0428–0.0429（约 1147）。
// 取 0.065（空闲时约合 1740 GFLOPS）：naive 低于阈值 1.52×，16×16 变体高于阈值 1.77×，参考答案 2.66×。
// 抗干扰实测（同卡另有 2–3 个进程）：循环跑本用例的副本——比值与空闲基本相同；持续不停发射参考 GEMM（重负载）——参考答案
// 0.078–0.110（全过；绝对阈值 2000 时它只有约 1300 GFLOPS、全挂），16×16 变体 0.060–0.087（单次 21 中 17 过，判分器另有 3 次重试），
// naive 0.035–0.052（全挂）。满载时 GEMM 掉速比 FMA 微基准多，比值偏低；用 16 KiB LDS 的基准则会反过来让 naive 越线，故不用。详见 docs/reviews/lab1.md。
constexpr double kTiledRatio = 0.065;
PERF_CASE(tiled_gflops, 4) {
  const int M = 2048, N = 2048, K = 2048;
  auto ha = hk::random_floats(size_t(M) * K, 5), hb = hk::random_floats(size_t(K) * N, 6);
  hk::GuardedBuffer<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
  A.upload(ha); B.upload(hb);
  auto launch = [&] { launch_gemm_tiled(A.get(), B.get(), C.get(), M, N, K, nullptr); };
  hk::time_best_ms(launch, 2, 1);
  double peak = hk::fma_peak_gflops(1, 1), ms = 1e30;
  for (int r = 0; r < 10; ++r) {  // 交替：微基准、学生 kernel、微基准、学生 kernel……
    peak = std::max(peak, hk::fma_peak_gflops(0, 1));
    ms = std::min(ms, hk::time_best_ms(launch, 0, 1));
  }
  R.metric = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
  R.unit = "GFLOPS";
  if (!C.guards_intact(R)) return;
  if (A.download() != ha || B.download() != hb) { hk::fail(R, "输入被修改了"); return; }
  auto hc = C.download();
  // 校验 1：每行和 C·1 与 A·(B·1)、每列和 1ᵀC 与 (1ᵀA)·B 比较，覆盖全部 M×N 个元素（未写的元素是 NaN，必然暴露）。
  // 误差界：逐元素允许 |e_ij| ≤ 1e-5·Σ_k|a_ik·b_kj| + 1e-6（与下面抽查、功能用例相同），对一行/一列求和得
  //   行 i：1e-5·Σ_k |a_ik|·(Σ_j |b_kj|) + N·1e-6；列 j：1e-5·Σ_k (Σ_i |a_ik|)·|b_kj| + M·1e-6，
  // 即只要每个元素满足逐元素误差界，行/列和就一定满足——不会比逐元素检查更严格（CPU 端用 double 求和，其误差可忽略）。
  {
    std::vector<double> b1(K, 0), b1abs(K, 0), a1(K, 0), a1abs(K, 0);
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j) { double b = hb[size_t(k) * N + j]; b1[k] += b; b1abs[k] += std::fabs(b); }
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k) { double a = ha[size_t(i) * K + k]; a1[k] += a; a1abs[k] += std::fabs(a); }
    for (int i = 0; i < M; ++i) {
      double ref = 0, mag = 0, got = 0;
      for (int k = 0; k < K; ++k) { double a = ha[size_t(i) * K + k]; ref += a * b1[k]; mag += std::fabs(a) * b1abs[k]; }
      for (int j = 0; j < N; ++j) got += hc[size_t(i) * N + j];
      if (!(std::fabs(got - ref) <= 1e-5 * mag + N * 1e-6)) {
        hk::fail(R, "大矩阵结果错误：第 %d 行之和期望 %.7g，得到 %.7g", i, ref, got);
        return;
      }
    }
    std::vector<double> colsum(N, 0);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) colsum[j] += hc[size_t(i) * N + j];
    for (int j = 0; j < N; ++j) {
      double ref = 0, mag = 0;
      for (int k = 0; k < K; ++k) { double b = hb[size_t(k) * N + j]; ref += a1[k] * b; mag += a1abs[k] * std::fabs(b); }
      if (!(std::fabs(colsum[j] - ref) <= 1e-5 * mag + M * 1e-6)) {
        hk::fail(R, "大矩阵结果错误：第 %d 列之和期望 %.7g，得到 %.7g", j, ref, colsum[j]);
        return;
      }
    }
  }
  // 校验 2：每个 32×32 输出块里随机抽 2 个元素做逐元素检查
  hk::Rng rng(3);
  for (int ti = 0; ti < M; ti += 32)
    for (int tj = 0; tj < N; tj += 32)
      for (int t = 0; t < 2; ++t) {
        int i = std::min(M - 1, ti + rng.randint(0, 31)), j = std::min(N - 1, tj + rng.randint(0, 31));
        double ref = 0, mag = 0;
        for (int k = 0; k < K; ++k) { double p = double(ha[size_t(i) * K + k]) * hb[size_t(k) * N + j]; ref += p; mag += std::fabs(p); }
        if (!(std::fabs(hc[size_t(i) * N + j] - ref) <= 1e-5 * mag + 1e-6)) {
          hk::fail(R, "大矩阵结果错误：C[%d][%d] 期望 %.7g，得到 %.7g", i, j, ref, double(hc[size_t(i) * N + j]));
          return;
        }
      }
  if (R.metric < kTiledRatio * peak)
    hk::fail(R, "%.0f GFLOPS，低于目标 %.0f GFLOPS（= %.1f%% × 本次实测 fp32 FMA 吞吐 %.0f GFLOPS）", R.metric,
             kTiledRatio * peak, kTiledRatio * 100, peak);
}
HK_MAIN()

#include "harness.hpp"
#include STUDENT_SRC

using Fn = void (*)(const float*, const float*, float*, int, int, int, hipStream_t);

static void run(hk::Result& R, Fn fn, const char* name, int M, int N, int K, uint64_t seed) {
  auto ha = hk::random_floats(size_t(M) * K, seed), hb = hk::random_floats(size_t(K) * N, seed + 1);
  hk::GuardedBuffer<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
  A.upload(ha); B.upload(hb); C.fill(NAN);
  fn(A.get(), B.get(), C.get(), M, N, K, nullptr);
  auto hc = C.download();
  // 逐元素误差界：|err| ≤ 1e-5 · Σ_k |a_ik·b_kj| + 1e-6（与求和顺序无关的 fp32 误差上界的宽松版本）
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

static const int kShapes[][3] = {{1, 1, 1}, {17, 33, 65}, {64, 64, 64}, {128, 96, 200}, {500, 300, 257}, {512, 512, 512}};

TEST_CASE(naive_small, 1) { run(R, launch_gemm_naive, "naive", 4, 5, 6, 1); }
TEST_CASE(naive_all_shapes, 3) {
  for (auto& s : kShapes) { run(R, launch_gemm_naive, "naive", s[0], s[1], s[2], 2); if (!R.passed) return; }
}
TEST_CASE(tiled_aligned, 2) { run(R, launch_gemm_tiled, "tiled", 128, 128, 128, 3); }
TEST_CASE(tiled_all_shapes, 4) {
  for (auto& s : kShapes) { run(R, launch_gemm_tiled, "tiled", s[0], s[1], s[2], 4); if (!R.passed) return; }
}

// 校准（gfx942 空闲卡，5 次）：参考答案(32×32 tile, 32×8 block, 每线程 4 输出) ≈ 4636 GFLOPS（最小 4635.6），
// naive(m3) ≈ 1149 GFLOPS（最大 1148.8），S/N≈4.0；几何平均 2307；按审查要求下调为 2000（对 naive 1.74×、16×16 变体 1.55×、参考答案 2.3× 余量）。
// 合理自写变体：16×16 tile/16×16 block/每线程 1 输出 ≈ 3107；32×32 tile 无 padding ≈ 4553；均高于阈值。
constexpr double kTiledGflops = 2000.0;
PERF_CASE(tiled_gflops, 4) {
  const int M = 2048, N = 2048, K = 2048;
  auto ha = hk::random_floats(size_t(M) * K, 5), hb = hk::random_floats(size_t(K) * N, 6);
  hk::GuardedBuffer<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
  A.upload(ha); B.upload(hb);
  double ms = hk::time_best_ms([&] { launch_gemm_tiled(A.get(), B.get(), C.get(), M, N, K, nullptr); }, 2, 10);
  R.metric = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
  R.unit = "GFLOPS";
  auto hc = C.download();
  hk::Rng rng(3);
  for (int t = 0; t < 200; ++t) {  // 抽查 200 个元素
    int i = rng.randint(0, M - 1), j = rng.randint(0, N - 1);
    double ref = 0, mag = 0;
    for (int k = 0; k < K; ++k) { double p = double(ha[size_t(i) * K + k]) * hb[size_t(k) * N + j]; ref += p; mag += std::fabs(p); }
    if (!(std::fabs(hc[size_t(i) * N + j] - ref) <= 1e-5 * mag + 1e-6)) { hk::fail(R, "大矩阵结果错误：C[%d][%d]", i, j); return; }
  }
  if (R.metric < kTiledGflops)
    hk::fail(R, "%.0f GFLOPS，低于目标 %.0f GFLOPS（若机器繁忙，用 rocm-smi 确认后重试）", R.metric, kTiledGflops);
}
HK_MAIN()

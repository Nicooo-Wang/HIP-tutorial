#include "harness.hpp"
#include STUDENT_SRC

static void run(hk::Result& R, int rows, int cols, uint64_t seed) {
  size_t n = size_t(rows) * cols;
  auto h = hk::random_floats(n, seed);
  hk::GuardedBuffer<float> in(n), out(n);
  in.upload(h); out.fill(NAN);
  launch_ck_transpose(in.get(), out.get(), rows, cols, nullptr);
  std::vector<double> ref(n);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[size_t(c) * rows + r] = h[size_t(r) * cols + c];
  char w[96]; std::snprintf(w, sizeof(w), "rows=%d cols=%d（out 为 cols×rows）", rows, cols);
  if (!hk::check_close(R, out.download(), ref, 0, 0, w, rows)) return;
  if (!out.guards_intact(R)) return;
  if (in.download() != h) hk::fail(R, "输入被修改了");
}

TEST_CASE(one_tile, 2)       { run(R, 64, 64, 1); }
TEST_CASE(multi_tile, 2)     { run(R, 128, 256, 2); run(R, 256, 128, 3); }
TEST_CASE(ragged, 4)         { run(R, 33, 31, 4); run(R, 100, 70, 5); run(R, 1, 1, 6); run(R, 1000, 17, 7); run(R, 17, 1000, 8); }
TEST_CASE(large, 1)          { run(R, 2048, 1024, 9); }

// 校准（gfx942，空闲 GPU 0/1，每种 3–4 次，墙钟计时，比值 = 有效带宽 / 同进程 D2D 拷贝带宽）：
//   参考解（64×64 tile、LDS padding、4 wave）0.54–0.57；
//   合理的替代设计：64×64 无 LDS padding 0.52–0.54，64×64 向量宽度 2 0.65–0.67，64×64 行交错分布 0.55–0.57，
//   32×32 tile 0.455–0.465（最慢的正确设计）。
//   候选慢 mutant m4（每 block 只有 1 个 wave）实测 0.51–0.52，与参考解几乎无差别（S/N ≈ 1.04 < 2.25），
//   不能用于区分，已从 mutants 中删除；按兜底规则阈值 = floor2(S_min / 1.67) = floor2(0.323) = 0.32，
//   再因 32×32 替代设计只有 0.455，为保证其 ≥ 1.5× 余量下调为 0.30（参考解余量 ≥ 1.8×，32×32 ≥ 1.5×）。
//   本用例的价值是兜底（排除明显低效/错误的实现，如逐元素标量搬运），区分度主要由正确性用例承担。
constexpr double kCkTransposeRatio = 0.30;
PERF_CASE(bandwidth, 3) {
  const int rows = 8192, cols = 8192;
  size_t n = size_t(rows) * cols, bytes = n * sizeof(float);
  auto h = hk::random_floats(n, 10);
  hk::GuardedBuffer<float> in(n), out(n);
  in.upload(h);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { launch_ck_transpose(in.get(), out.get(), rows, cols, nullptr); });
  R.metric = 2.0 * bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  auto o = out.download();
  hk::Rng rng(12);
  for (int k = 0; k < 100000; ++k) {
    int r = rng.randint(0, rows - 1), c = rng.randint(0, cols - 1);
    if (o[size_t(c) * rows + r] != h[size_t(r) * cols + c]) { hk::fail(R, "大矩阵结果错误（行 %d 列 %d）", r, c); return; }
  }
  if (R.metric < kCkTransposeRatio * copy) {
    hk::fail(R, "带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kCkTransposeRatio * copy, kCkTransposeRatio * 100, copy);
  } else {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "本次实测拷贝带宽 %.0f GB/s，比值 %.2f（目标 ≥ %.2f）", copy, R.metric / copy, kCkTransposeRatio);
    R.msg = buf;
  }
}
HK_MAIN()

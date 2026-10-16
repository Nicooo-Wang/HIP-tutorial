#include "harness.hpp"
#include STUDENT_SRC

using LaunchFn = void (*)(const float*, float*, int, int, hipStream_t);

static void run(hk::Result& R, LaunchFn fn, const char* name, int rows, int cols, uint64_t seed) {
  size_t n = size_t(rows) * cols;
  auto h = hk::random_floats(n, seed);
  hk::GuardedBuffer<float> in(n), out(n);
  in.upload(h); out.fill(NAN);
  fn(in.get(), out.get(), rows, cols, nullptr);
  std::vector<double> ref(n);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[size_t(c) * rows + r] = h[size_t(r) * cols + c];
  char w[96]; std::snprintf(w, sizeof(w), "%s rows=%d cols=%d（out 为 cols×rows）", name, rows, cols);
  if (!hk::check_close(R, out.download(), ref, 0, 0, w, rows)) return;
  if (!out.guards_intact(R)) return;
  if (in.download() != h) hk::fail(R, "%s: 输入被修改了", name);
}

static const int kShapes[][2] = {{1, 1}, {32, 32}, {33, 31}, {1000, 17}, {17, 1000}, {1024, 2048}};

TEST_CASE(naive_square, 1) { run(R, launch_transpose_naive, "naive", 32, 32, 1); }
TEST_CASE(naive_all_shapes, 2) {
  for (auto& s : kShapes) { run(R, launch_transpose_naive, "naive", s[0], s[1], 2); if (!R.passed) return; }
}
TEST_CASE(lds_square, 1) { run(R, launch_transpose_lds, "lds", 64, 64, 3); }
TEST_CASE(lds_all_shapes, 3) {
  for (auto& s : kShapes) { run(R, launch_transpose_lds, "lds", s[0], s[1], 4); if (!R.passed) return; }
}

// 性能：LDS 版本的有效带宽（读 + 写）需达到同进程 D2D 拷贝带宽的 kLdsRatio。
// kLdsRatio 已校准：解约 0.34、naive 约 0.064（各 ≥2.2× 余量）。
constexpr double kLdsRatio = 0.14;
PERF_CASE(lds_bandwidth, 3) {
  const int rows = 8192, cols = 8192;
  size_t n = size_t(rows) * cols, bytes = n * sizeof(float);
  auto h = hk::random_floats(n, 5);
  hk::GuardedBuffer<float> in(n), out(n);
  in.upload(h);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { launch_transpose_lds(in.get(), out.get(), rows, cols, nullptr); });
  R.metric = 2.0 * bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  // 同时抽查结果，防止"什么都不做所以很快"
  auto o = out.download();
  hk::Rng rng(11);
  for (int k = 0; k < 100000; ++k) {
    int r = rng.randint(0, rows - 1), c = rng.randint(0, cols - 1);
    if (o[size_t(c) * rows + r] != h[size_t(r) * cols + c]) { hk::fail(R, "大矩阵结果错误（行 %d 列 %d）", r, c); return; }
  }
  if (R.metric < kLdsRatio * copy)
    hk::fail(R, "带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kLdsRatio * copy, kLdsRatio * 100, copy);
}
HK_MAIN()

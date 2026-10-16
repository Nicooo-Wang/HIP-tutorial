#include "harness.hpp"
#include STUDENT_SRC

static void run(hk::Result& R, int rows, int cols, uint64_t seed) {
  size_t n = size_t(rows) * cols;
  auto h = hk::random_floats(n, seed);
  hk::GuardedBuffer<float> in(n), out(rows);
  in.upload(h); out.fill(NAN);
  launch_ck_row_sum(in.get(), out.get(), rows, cols, nullptr);
  std::vector<double> ref(rows);
  double tol = 0;
  for (int r = 0; r < rows; ++r) {
    double s = 0, mag = 0;
    for (int c = 0; c < cols; ++c) { s += h[size_t(r) * cols + c]; mag += std::fabs(h[size_t(r) * cols + c]); }
    ref[r] = s; tol = std::max(tol, 1e-5 * mag + 1e-6);
  }
  char w[64]; std::snprintf(w, sizeof(w), "rows=%d cols=%d", rows, cols);
  if (!hk::check_close(R, out.download(), ref, tol, 0, w)) return;
  if (!out.guards_intact(R)) return;
  if (in.download() != h) hk::fail(R, "输入被修改了");
}

// 全 1 输入的精确检查：每行之和应精确等于 cols（抓住漏列/重复列）
static void ones(hk::Result& R, int rows, int cols) {
  size_t n = size_t(rows) * cols;
  hk::GuardedBuffer<float> in(n), out(rows);
  in.fill(1.0f); out.fill(NAN);
  launch_ck_row_sum(in.get(), out.get(), rows, cols, nullptr);
  char w[64]; std::snprintf(w, sizeof(w), "全 1 rows=%d cols=%d", rows, cols);
  if (hk::check_close(R, out.download(), std::vector<double>(rows, double(cols)), 0, 0, w)) out.guards_intact(R);
}

TEST_CASE(aligned_small, 2)     { ones(R, 8, 256); run(R, 8, 256, 1); }
TEST_CASE(many_col_tiles, 2)    { ones(R, 16, 4096); run(R, 24, 1024 + 256, 2); }
TEST_CASE(ragged_cols, 3)       { ones(R, 5, 31); ones(R, 3, 257); run(R, 33, 31, 3); run(R, 100, 1001, 4); run(R, 7, 4099, 5); }
TEST_CASE(ragged_rows, 2)       { ones(R, 1, 256); ones(R, 13, 512); run(R, 1, 1, 6); run(R, 1001, 64, 7); }
TEST_CASE(large, 1)             { run(R, 4096, 4096, 8); }

// 校准（gfx942，空闲 GPU 0/1，每种 3–4 次，墙钟计时；比值 = 读带宽（只算读入字节）/ 同进程 D2D 拷贝带宽。
// 8192×8192 fp32 全 1 输入，读带宽可以高于“读+写”的拷贝带宽，所以比值 > 1 是正常的）：
//   参考解（4 wave、每 block 8 行、向量 4）1.66–1.77；
//   合理的替代设计：每 warp 1 行（TM=4）1.43–1.52，每 warp 4 行（TM=16）1.76–1.84，两个 tile 一次迭代 1.65–1.75，
//   1 个 wave/block 1.60–1.70，16 个 wave/block 1.53–1.61，向量宽度 2（cols%2==0 时）1.22–1.35（最慢的合理设计）。
//   候选慢 mutant m4（总是向量 1）实测 1.04–1.13，与参考解 S/N ≈ 1.47 < 2.25，无法同时留出 1.5× 余量，
//   按兜底规则已从 mutants 中删除；阈值取 0.80（最慢的合理替代 1.22 / 0.80 ≈ 1.5×），
//   本用例只是“不至于慢得离谱”的下限，不检查是否向量化（见 README“判分覆盖不到”）。
constexpr double kRowSumRatio = 0.80;
PERF_CASE(bandwidth_aligned, 3) {
  const int rows = 8192, cols = 8192;  // cols % 4 == 0：应走向量化路径
  size_t n = size_t(rows) * cols, bytes = n * sizeof(float);
  hk::GuardedBuffer<float> in(n), out(rows);
  in.fill(1.0f);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { launch_ck_row_sum(in.get(), out.get(), rows, cols, nullptr); });
  R.metric = bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  auto o = out.download();
  for (int r = 0; r < rows; ++r) if (o[r] != float(cols)) { hk::fail(R, "结果错误 out[%d]=%g", r, o[r]); return; }
  if (R.metric < kRowSumRatio * copy) {
    hk::fail(R, "读带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）。cols 是 4 的倍数时是否用了向量化配置？",
             R.metric, kRowSumRatio * copy, kRowSumRatio * 100, copy);
  } else {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "本次实测拷贝带宽 %.0f GB/s，读带宽 %.0f GB/s，比值 %.2f（目标 ≥ %.2f）", copy, R.metric, R.metric / copy, kRowSumRatio);
    R.msg = buf;
  }
}
HK_MAIN()

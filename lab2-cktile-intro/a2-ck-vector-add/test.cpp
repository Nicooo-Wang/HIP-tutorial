#include "harness.hpp"
#include STUDENT_SRC

static void run(hk::Result& R, int n, uint64_t seed) {
  auto ha = hk::random_floats(n, seed), hb = hk::random_floats(n, seed + 1);
  hk::GuardedBuffer<float> a(n), b(n), c(n);
  a.upload(ha); b.upload(hb); c.fill(NAN);
  launch_ck_vector_add(a.get(), b.get(), c.get(), n, nullptr);
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = double(ha[i] + hb[i]);
  char w[64]; std::snprintf(w, sizeof(w), "n=%d", n);
  if (!hk::check_close(R, c.download(), ref, 0, 0, w)) return;
  if (!c.guards_intact(R)) return;
  if (a.download() != ha || b.download() != hb) hk::fail(R, "输入被修改了");
}

TEST_CASE(one_tile, 2)        { run(R, 1024, 1); }
TEST_CASE(many_tiles, 2)      { run(R, 1024 * 37, 2); }
TEST_CASE(tiny, 2)            { run(R, 1, 3); run(R, 3, 4); run(R, 64, 5); }
TEST_CASE(ragged, 3)          { run(R, 1023, 6); run(R, 4097, 7); run(R, 1000003, 8); }
TEST_CASE(large, 1)           { run(R, (1 << 24) + 5, 9); }

// 校准（比值 = 带宽 / 同进程实测 D2D 拷贝带宽，GPU 1、5 各 4 次）：参考答案 S_min = 1.45；单 block 慢版本 N_max = 0.013；
// 其他合理写法最低：kVec=1 为 0.84，kVec=2 为 1.17，kWarps=1/2 超过 1.8。0.55 使 kVec=1 留 1.5× 余量、参考答案 2.6×、慢版本 42×。
constexpr double kAddRatio = 0.55;
PERF_CASE(bandwidth, 2) {
  int n = 1 << 26;
  size_t bytes = size_t(n) * sizeof(float);
  hk::GuardedBuffer<float> a(n), b(n), c(n);
  a.fill(1.0f); b.fill(2.0f);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { launch_ck_vector_add(a.get(), b.get(), c.get(), n, nullptr); });
  R.metric = 3.0 * bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  auto h = c.download();
  for (int i = 0; i < n; i += 9973) if (h[i] != 3.0f) { hk::fail(R, "结果错误 c[%d]=%g", i, h[i]); return; }
  if (h[n - 1] != 3.0f) { hk::fail(R, "结果错误 c[n-1]"); return; }
  if (R.metric < kAddRatio * copy) {
    hk::fail(R, "带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kAddRatio * copy, kAddRatio * 100, copy);
  } else {
    // 通过时也留下基准，供 make run 打印（书面问题 1 的表格要用）；make grade 只在失败时显示 msg
    char buf[160];
    std::snprintf(buf, sizeof(buf), "本次实测拷贝带宽 %.0f GB/s，比值 %.2f（目标 ≥ %.2f）", copy, R.metric / copy, kAddRatio);
    R.msg = buf;
  }
}
HK_MAIN()

#include "harness.hpp"
#include STUDENT_SRC

static void run_add(hk::Result& R, int n, uint64_t seed) {
  auto ha = hk::random_floats(n, seed), hb = hk::random_floats(n, seed + 1);
  hk::GuardedBuffer<float> a(n), b(n), c(n);
  a.upload(ha); b.upload(hb); c.fill(NAN);
  launch_vector_add(a.get(), b.get(), c.get(), n, nullptr);
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = double(ha[i] + hb[i]);
  char w[64]; std::snprintf(w, sizeof(w), "vector_add n=%d", n);
  if (!hk::check_close(R, c.download(), ref, 0, 0, w)) return;
  c.guards_intact(R);
  if (a.download() != ha || b.download() != hb) hk::fail(R, "输入 a/b 被修改了");
}

static void run_saxpy(hk::Result& R, int n, uint64_t seed) {
  const float alpha = 1.75f;
  auto hx = hk::random_floats(n, seed), hy = hk::random_floats(n, seed + 7);
  hk::GuardedBuffer<float> x(n), y(n);
  x.upload(hx); y.upload(hy);
  launch_saxpy(alpha, x.get(), y.get(), n, nullptr);
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = double(std::fmaf(alpha, hx[i], hy[i]));
  char w[64]; std::snprintf(w, sizeof(w), "saxpy n=%d", n);
  if (hk::check_close(R, y.download(), ref, 1e-6, 1e-6, w)) y.guards_intact(R);
}

TEST_CASE(add_small, 1)        { run_add(R, 1, 1); run_add(R, 7, 2); }
TEST_CASE(add_block_exact, 1)  { run_add(R, 1024, 3); }
TEST_CASE(add_ragged, 2)       { run_add(R, 1023, 4); run_add(R, 100003, 5); }
TEST_CASE(add_large, 2)        { run_add(R, (1 << 24) + 1, 6); }
TEST_CASE(saxpy_fits_grid, 1)  { run_saxpy(R, kSaxpyGrid * 256, 7); }
TEST_CASE(saxpy_small, 1)      { run_saxpy(R, 5, 8); }
TEST_CASE(saxpy_needs_stride, 3) { run_saxpy(R, kSaxpyGrid * 256 * 37 + 11, 9); }
HK_MAIN()

#include "harness.hpp"
#include STUDENT_SRC

static void check_n(hk::Result& R, int n) {
  hk::GuardedBuffer<int> out(n);
  out.fill(-1);
  launch_fill_ids(out.get(), n, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = i;
  char what[64];
  std::snprintf(what, sizeof(what), "n=%d", n);
  if (hk::check_equal(R, out.download(), ref, what)) out.guards_intact(R);
}

TEST_CASE(one_block_exact, 1) { check_n(R, 256); }
TEST_CASE(single_element, 1) { check_n(R, 1); }
TEST_CASE(not_multiple_of_block, 2) { check_n(R, 1000); check_n(R, 257); }
TEST_CASE(many_blocks, 2) { check_n(R, (1 << 22) + 3); }
HK_MAIN()

#include "harness.hpp"
#include STUDENT_SRC
#include <chrono>
#include <thread>

static const char* mode() { const char* m = std::getenv("FAKE_MODE"); return m ? m : ""; }

TEST_CASE(value_is_42, 3) {
  if (student_value() != 42) hk::fail(R, "期望 42，得到 %d", student_value());
}
TEST_CASE(maybe_crash, 1) {
  if (std::string(mode()) == "crash") std::abort();
}
TEST_CASE(maybe_hang, 1) {
  if (std::string(mode()) == "hang") for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}
PERF_CASE(maybe_flaky_perf, 1) {
  // FAKE_MODE=flaky:<file> —— 第一次运行失败、之后成功，用于测试性能用例重试
  std::string m = mode();
  R.unit = "x";
  if (m.rfind("flaky:", 0) == 0) {
    std::string f = m.substr(6);
    FILE* fp = std::fopen(f.c_str(), "r");
    if (!fp) { fp = std::fopen(f.c_str(), "w"); if (fp) std::fclose(fp); R.metric = 1; hk::fail(R, "首次慢"); return; }
    std::fclose(fp);
  }
  R.metric = 2;
}
HK_MAIN()

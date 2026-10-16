// harness.hpp —— 本课程所有 test.cpp 共用的测试框架。学生无需修改。
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                              \
  do {                                                                               \
    hipError_t hk_err_ = (expr);                                                     \
    if (hk_err_ != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP 错误: %s\n  位置 %s:%d\n  原因 %s\n", #expr, __FILE__, \
                   __LINE__, hipGetErrorString(hk_err_));                            \
      std::exit(2);                                                                  \
    }                                                                                \
  } while (0)

namespace hk {

struct Result {
  bool passed = true;
  std::string msg;
  double metric = 0.0;
  std::string unit;
};

inline void fail(Result& R, const char* fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int len = std::vsnprintf(nullptr, 0, fmt, ap);
  va_end(ap);
  std::string buf(len > 0 ? size_t(len) : 0, '\0');
  if (len > 0) std::vsnprintf(&buf[0], size_t(len) + 1, fmt, ap2);
  va_end(ap2);
  R.passed = false;
  if (!R.msg.empty()) R.msg += "\n";
  R.msg += buf;
}

struct Case {
  std::string name;
  int points;
  bool perf;
  std::function<void(Result&)> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* n, int p, bool perf, void (*f)(Result&)) {
    registry().push_back({n, p, perf, f});
  }
};

#define HK_DEFINE_CASE(name, points, perf)                               \
  static void hk_case_##name(hk::Result& R);                             \
  static hk::Registrar hk_reg_##name(#name, points, perf, hk_case_##name); \
  static void hk_case_##name([[maybe_unused]] hk::Result& R)
#define TEST_CASE(name, points) HK_DEFINE_CASE(name, points, false)
#define PERF_CASE(name, points) HK_DEFINE_CASE(name, points, true)

// ---------------- 随机数（splitmix64，跨平台可复现） ----------------
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * float((next() >> 40) * (1.0 / 16777216.0));
  }
  int randint(int lo, int hi) {
    if (hi < lo) std::swap(lo, hi);
    return lo + int(next() % uint64_t(hi - lo + 1)); }
};

inline std::vector<float> random_floats(size_t n, uint64_t seed, float lo = -1.f, float hi = 1.f) {
  Rng r(seed);
  std::vector<float> v(n);
  for (auto& x : v) x = r.uniform(lo, hi);
  return v;
}

// 全局（与模板参数无关）的哨兵序号：任意类型的相邻缓冲区哨兵都不同。
// 哨兵字节的选取：把重复字节按 float32 解码，指数位必须很大（字节取 0x60–0x7F 或 0xE0–0xFF，数值约 ±1e19 ~ 3e38，
// 0xFF 为 NaN）。否则像 0xA5（≈-2e-16）、0x1E 这类哨兵读成 float 接近 0，学生 kernel 越界读入后乘以它几乎不改变结果，
// 漏掉的边界错误就能蒙混过关；取巨大值后任何越界读都会明显污染结果。
inline unsigned char next_guard_pattern() {
  static const unsigned char kPatterns[] = {0x6A, 0xE5, 0x75, 0xF2, 0x7F, 0xEA, 0x63, 0xFB,
                                            0x6C, 0xE1, 0x79, 0xF6, 0x66, 0xED, 0x7B, 0xFD};
  static int counter = 0;
  return kPatterns[counter++ % 16];
}

// ---------------- 带哨兵的设备缓冲区：检测越界写 ----------------
template <class T>
struct GuardedBuffer {
  static constexpr size_t kGuardBytes = 4096;
  size_t n;
  char* raw = nullptr;
  // 每个缓冲区用不同的哨兵字节：防止"越界读到输入哨兵、再越界写进输出哨兵"恰好写回相同值而漏检。
  unsigned char pattern;

  explicit GuardedBuffer(size_t n_) : n(n_) {
    pattern = next_guard_pattern();
    HIP_CHECK(hipMalloc(&raw, bytes() + 2 * kGuardBytes));
    HIP_CHECK(hipMemset(raw, pattern, bytes() + 2 * kGuardBytes));
  }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() { (void)hipFree(raw); }

  size_t bytes() const { return n * sizeof(T); }
  T* get() { return reinterpret_cast<T*>(raw + kGuardBytes); }
  void upload(const std::vector<T>& h) {
    HIP_CHECK(hipMemcpy(get(), h.data(), bytes(), hipMemcpyHostToDevice));
  }
  std::vector<T> download() {
    std::vector<T> h(n);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h.data(), get(), bytes(), hipMemcpyDeviceToHost));
    return h;
  }
  void fill(T v) { upload(std::vector<T>(n, v)); }
  bool guards_intact(Result& R) {
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<unsigned char> g(kGuardBytes);
    const char* where[2] = {"之前", "之后"};
    const char* ptrs[2] = {raw, raw + kGuardBytes + bytes()};
    for (int s = 0; s < 2; ++s) {
      HIP_CHECK(hipMemcpy(g.data(), ptrs[s], kGuardBytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < kGuardBytes; ++i)
        if (g[i] != pattern) {
          fail(R, "越界写：缓冲区（%zu 个元素）%s的哨兵区被改写（距边界 %zu 字节）", n, where[s],
               s == 0 ? kGuardBytes - i : i + 1);
          return false;
        }
    }
    return true;
  }
};

// ---------------- 结果比较 ----------------
inline std::string fmt_index(size_t i, int cols) {
  char b[64];
  if (cols > 0) std::snprintf(b, sizeof(b), "[%zu]（行 %zu, 列 %zu）", i, i / cols, i % cols);
  else std::snprintf(b, sizeof(b), "[%zu]", i);
  return b;
}

inline bool check_close(Result& R, const std::vector<float>& got, const std::vector<double>& ref,
                        double atol, double rtol, const char* what, int cols = 0) {
  if (got.size() != ref.size()) {
    fail(R, "%s: 长度不符（得到 %zu，期望 %zu）", what, got.size(), ref.size());
    return false;
  }
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double g = got[i], e = ref[i];
    bool ok;
    if (std::isnan(e)) ok = std::isnan(g);
    else if (std::isinf(e)) ok = (g == e);
    else ok = std::isfinite(g) && std::fabs(g - e) <= atol + rtol * std::fabs(e);
    if (!ok && bad++ == 0) first = i;
  }
  if (bad) {
    fail(R, "%s: %zu/%zu 个元素错误，首个错误在 %s：期望 %.7g，得到 %.7g", what, bad, got.size(),
         fmt_index(first, cols).c_str(), ref[first], double(got[first]));
    return false;
  }
  return true;
}

template <class I>
inline bool check_equal(Result& R, const std::vector<I>& got, const std::vector<I>& ref,
                        const char* what, int cols = 0) {
  if (got.size() != ref.size()) {
    fail(R, "%s: 长度不符（得到 %zu，期望 %zu）", what, got.size(), ref.size());
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != ref[i]) {
      size_t bad = 0;
      for (size_t j = 0; j < got.size(); ++j) bad += got[j] != ref[j];
      fail(R, "%s: %zu/%zu 个元素错误，首个错误在 %s：期望 %lld，得到 %lld", what, bad, got.size(),
           fmt_index(i, cols).c_str(), (long long)ref[i], (long long)got[i]);
      return false;
    }
  return true;
}

// ---------------- 计时 ----------------
template <class F>
inline double time_best_ms(F&& launch, int warmup = 3, int iters = 20) {
  for (int i = 0; i < warmup; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    best = std::min(best, double(ms));
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return best;
}

// 同进程实测 D2D 拷贝带宽（读 + 写字节 / 时间），作为内存类性能用例的基准。
inline double copy_bandwidth_gbs(size_t bytes) {
  void *s, *d;
  HIP_CHECK(hipMalloc(&s, bytes));
  HIP_CHECK(hipMalloc(&d, bytes));
  HIP_CHECK(hipMemset(s, 1, bytes));
  double ms = time_best_ms([&] { HIP_CHECK(hipMemcpyAsync(d, s, bytes, hipMemcpyDeviceToDevice)); });
  HIP_CHECK(hipFree(s));
  HIP_CHECK(hipFree(d));
  return 2.0 * bytes / (ms * 1e-3) / 1e9;
}

// ---------------- 进程入口 ----------------
inline std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
    else o += char(c);
  }
  return o;
}

inline void run_one(const Case& c, Result& R) {
  c.fn(R);
  hipError_t e = hipDeviceSynchronize();
  if (e == hipSuccess) e = hipGetLastError();
  if (e != hipSuccess) fail(R, "kernel 运行错误: %s", hipGetErrorString(e));
}

inline int run_main(int argc, char** argv) {
  auto& cases = registry();
  if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
    std::printf("[");
    for (size_t i = 0; i < cases.size(); ++i)
      std::printf("%s{\"name\":\"%s\",\"points\":%d,\"perf\":%s}", i ? "," : "", cases[i].name.c_str(),
                  cases[i].points, cases[i].perf ? "true" : "false");
    std::printf("]\n");
    return 0;
  }
  if (argc >= 3 && std::strcmp(argv[1], "--case") == 0) {
    for (auto& c : cases)
      if (c.name == argv[2]) {
        Result R;
        run_one(c, R);
        if (!std::isfinite(R.metric)) {
          R.metric = 0.0;
          if (!R.msg.empty()) R.msg += "\n";
          R.msg += "注意: metric 不是有限值，已置 0";
        }
        std::fflush(stdout);
        std::printf("\n@@RESULT {\"name\":\"%s\",\"passed\":%s,\"msg\":\"%s\",\"metric\":%.6g,\"unit\":\"%s\"}\n",
                    c.name.c_str(), R.passed ? "true" : "false", json_escape(R.msg).c_str(), R.metric,
                    json_escape(R.unit).c_str());
        return 0;
      }
    std::fprintf(stderr, "未知用例: %s\n", argv[2]);
    return 3;
  }
  int pass = 0, total = 0;
  if (argc >= 2) {
    bool found = false;
    for (auto& c : cases) found |= (c.name == argv[1]);
    if (!found) {
      std::fprintf(stderr, "未知用例: %s\n", argv[1]);
      return 3;
    }
  }
  for (auto& c : cases) {
    if (argc >= 2 && c.name != argv[1]) continue;
    Result R;
    run_one(c, R);
    ++total;
    pass += R.passed;
    std::printf("%-40s %s", c.name.c_str(), R.passed ? "OK" : "FAIL");
    if (c.perf) std::printf("  (%.1f %s)", R.metric, R.unit.c_str());
    std::printf("\n");
    if (!R.msg.empty()) std::printf("    %s\n", R.msg.c_str());
  }
  std::printf("通过 %d/%d\n", pass, total);
  return pass == total ? 0 : 1;
}

}  // namespace hk

#define HK_MAIN() \
  int main(int argc, char** argv) { return hk::run_main(argc, argv); }

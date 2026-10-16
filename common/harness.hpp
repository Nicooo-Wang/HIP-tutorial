// harness.hpp —— 本课程所有 test.cpp 共用的测试框架。学生无需修改。
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
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

// 哨兵（guard）值：每个缓冲区一个 32 位字，是带"载荷"的 quiet NaN：0x7FC00000 | payload，payload 由全局
// （与模板参数无关的）计数器生成，非 0 且相邻缓冲区各不相同。
// - NaN：越界读到的值只要参与运算（哪怕乘以 0），结果就是 NaN，必然被结果比较发现；
// - payload 各不相同且非 0：防止"越界读输入哨兵、再越界写进输出哨兵"恰好写回相同值，
//   也防止 GPU 运算产生的规范 NaN（0x7FC00000）写进哨兵区而漏检。
// 作为 int 读时它是一个约 2.1e9 的大正数，同样会明显污染结果。
inline uint32_t next_guard_word() {
  static uint32_t counter = 0;
  counter = counter % 0x3FFFFFu + 1;  // 1 .. 0x3FFFFF，循环
  return 0x7FC00000u | counter;
}

// ---------------- 带哨兵的设备缓冲区：检测越界写 / 让越界读产生 NaN ----------------
template <class T>
struct GuardedBuffer {
  // 每侧 256 KiB：足以覆盖"越界一整个 tile 行""稀疏地跳出几千个元素"这类错误。
  static constexpr size_t kGuardBytes = size_t(256) << 10;
  size_t n;
  char* raw = nullptr;
  // 本缓冲区的哨兵字（quiet NaN + 载荷）。整个分配（含数据区初值）都按 32 位字填充为该值。
  uint32_t pattern;

  explicit GuardedBuffer(size_t n_) : n(n_) {
    pattern = next_guard_word();
    HIP_CHECK(hipMalloc(&raw, total_bytes()));
    HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(raw), int(pattern), total_bytes() / 4));
    HIP_CHECK(hipDeviceSynchronize());
  }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() { (void)hipFree(raw); }

  size_t bytes() const { return n * sizeof(T); }
  // 数据区按 4 字节向上取整，保证整个分配是 32 位字的整数倍
  size_t total_bytes() const { return 2 * kGuardBytes + (bytes() + 3) / 4 * 4; }
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
    // 两侧哨兵区：[0, kGuardBytes) 与 [kGuardBytes + bytes(), total_bytes())。
    // 整个分配按 32 位字填充，所以偏移 o 处的字节必须等于 pattern 的第 (o % 4) 个字节
    // ——等价于逐 32 位字比较，同时也覆盖了数据区末尾与字边界不对齐（如 2 字节类型、奇数个元素）的情形。
    unsigned char pb[4];
    std::memcpy(pb, &pattern, 4);
    const size_t tail0 = kGuardBytes + bytes();
    std::vector<unsigned char> g(total_bytes() - tail0);
    // 之前：从紧邻数据区处往前找，报告离边界最近的被改写字节
    HIP_CHECK(hipMemcpy(g.data(), raw, kGuardBytes, hipMemcpyDeviceToHost));
    for (size_t d = 1; d <= kGuardBytes; ++d)
      if (g[kGuardBytes - d] != pb[(kGuardBytes - d) % 4]) {
        fail(R, "越界写：缓冲区（%zu 个元素）之前的哨兵区被改写（距边界 %zu 字节）", n, d);
        return false;
      }
    // 之后
    HIP_CHECK(hipMemcpy(g.data(), raw + tail0, g.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < g.size(); ++i)
      if (g[i] != pb[(tail0 + i) % 4]) {
        fail(R, "越界写：缓冲区（%zu 个元素）之后的哨兵区被改写（距边界 %zu 字节）", n, i + 1);
        return false;
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
// 每次迭代计时 = hipDeviceSynchronize() 之后到 launch + hipDeviceSynchronize() 返回的主机墙钟（steady_clock），取最优。
// 用整卡同步夹住而不是默认流上的 event：即使 launch 函数把 kernel 发到别的（非阻塞）流上，也会被如实计时。
// 代价是每次多约 10–20 µs 的同步开销；性能用例的单次耗时都在亚毫秒～毫秒级以上，影响可忽略。
template <class F>
inline double time_best_ms(F&& launch, int warmup = 3, int iters = 20) {
  for (int i = 0; i < warmup; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  HIP_CHECK(hipGetLastError());
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

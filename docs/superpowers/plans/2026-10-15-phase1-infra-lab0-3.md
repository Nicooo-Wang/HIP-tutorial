# 第一期：基础设施 + Lab 0–3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立"Tutorial + Assignment + 自动判分"学习仓库的公共基础设施，并完成 Lab 0（环境）、Lab 1（HIP 基础）、Lab 2（C++ 速成 + CK-Tile 入门）、Lab 3（CK-Tile 核心概念）。

**Architecture:** 每个 Assignment 是一个学生编辑的 `.hip` 文件 + 一个 `test.cpp`（通过 `#include STUDENT_SRC` 引入学生文件）。`test.cpp` 基于 `common/harness.hpp` 注册测试用例；`common/grade.py` 编译、选空闲 GPU、**每个用例单独进程运行**（崩溃/卡死只影响该用例）、汇总分数。`common/verify.py` 用 `solutions/` 中的参考答案（必须全过）与 mutants（必须至少挂一项）校验判分器本身。

**Tech Stack:** ROCm 7.2 hipcc（`--offload-arch=gfx942 -std=c++20`）、CK-Tile 头文件（`/opt/rocm/include/ck_tile`）、Python 3.10 标准库、GNU Make、静态 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-10-15-hip-tutorial-design.md`

## Global Constraints

- 所有面向学习者的文字（Tutorial、README、TODO 注释、grade 输出）用中文；代码标识符用英文。
- 编译命令固定：`/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -I<repo>/common`。
- 依赖仅限 ROCm 自带工具 + Python 3 标准库（无 numpy、无 torch、无 gtest）。
- 机器共享：单个用例运行超时 60 s；问题规模单次显存占用 ≤ 2 GB；默认自动选最空闲 GPU，`HIP_VISIBLE_DEVICES` 已设置时尊重用户设置。
- 每个 Assignment 必须有：学生骨架（能编译、至少一个用例 FAIL）、`solutions/` 参考答案（全部 PASS）、≥2 个确定性失败的 mutant（放在 `solutions/<lab>/<aK>/mutants/`，**不放在学生目录**，否则泄露答案——这是对 spec §3 的修正，Task 1 同步更新 spec）。
- mutant 必须**确定性**失败（不得依赖竞态，例如"漏 `__syncthreads`"不能作为必挂 mutant）。
- 性能用例：计入分数；阈值必须经过校准——参考答案与"慢 mutant"之间至少各留 1.5× 余量；做不到则不设该性能用例并在审查报告中记录。内存带宽类用例以"同进程实测 `hipMemcpyDeviceToDevice` 带宽"为基准，自校准抗干扰。
- 挖空格式：`// TODO(aK-n): <中文说明>（讲义 §x.y）`，每个 TODO 后跟 `// >>> 你的代码 <<<` 区域。
- Tutorial：每 Lab 一个 `tutorial.html`，共用 `common/tutorial.css`、`common/tutorial.js`；浅色/深色模式；外部脚本仅 cdnjs 且固定版本。
- 硬件事实以本机实测为准：gfx942、80 CU、wavefront 64、workgroup 最大 1024 线程、LDS 64 KiB/workgroup。

## Review Focus

1. 学生代码崩溃（内存访问错误）或死循环：grade 应只把该用例记 FAIL（带"崩溃/超时"说明），其余用例照常判分——Task 2 的 `test_grade.py` 用 crash/hang 用例覆盖。
2. 学生代码越界写但结果区正确：哨兵检测必须判 FAIL——Task 1 harness 自测覆盖 `GuardedBuffer`。
3. 学生用 atomicAdd 累加到未清零的输出：test 预先把输出填成 NaN/垃圾值，必须判 FAIL——Lab 1 a3 reduction 测试覆盖。
4. 其他用户占用 GPU 导致性能用例误判：grade 在性能用例失败时自动重试（最多 3 次取最好），并在输出中显示当时 GPU 占用——Task 2 覆盖重试逻辑。
5. 学生骨架原样提交：必须能编译且得分 < 满分（不能编译失败导致看不到任何提示）——verify.py 对每个 Assignment 检查骨架。

## 文件结构

```
CLAUDE.md, README.md, .gitignore
common/harness.hpp      测试框架（用例注册、GuardedBuffer、比较、计时、JSON 输出）
common/harness_selftest.cpp  harness 自测
common/grade.py         判分驱动
common/verify.py        判分器校验
common/lab.mk           各 Lab Makefile 的公共规则
common/test_grade.py    grade.py / verify.py 的 unittest（含 fake 作业 fixtures）
common/fixtures/        fake 作业：good / crash / hang / oob
common/tutorial.css, common/tutorial.js
site/index.html
lab0-env/  lab1-hip-basics/  lab2-cktile-intro/  lab3-cktile-concepts/
  ├── tutorial.html, Makefile, answers.md
  └── aK-<name>/{<name>.hip, test.cpp, README.md}
solutions/<lab>/aK-<name>/{<name>.hip, mutants/*.hip}
docs/reviews/labN.md    审查报告
```

---
### Task 1: 仓库骨架 + 测试框架 `common/harness.hpp`

**Files:**
- Create: `CLAUDE.md`, `.gitignore`, `common/harness.hpp`, `common/harness_selftest.cpp`
- Modify: `docs/superpowers/specs/2026-10-15-hip-tutorial-design.md`（§3 mutants 位置改到 `solutions/<lab>/<aK>/mutants/`）

**Interfaces（后续所有 test.cpp 依赖，名字不可改）:**
- `HIP_CHECK(expr)`：失败打印并 `exit(2)`。
- `TEST_CASE(name, points) { ... }`、`PERF_CASE(name, points) { ... }`：函数体内可用 `hk::Result& R`。
- `HK_MAIN()`：放在 test.cpp 末尾，生成 `main`。
- `hk::Rng(seed)`：`.next()`、`.uniform(lo,hi)`、`.randint(lo,hi)`（闭区间）。
- `hk::random_floats(n, seed, lo=-1.f, hi=1.f) -> std::vector<float>`
- `hk::GuardedBuffer<T>(n)`：每个实例哨兵字节不同（`.pattern`）；`.get()`、`.upload(vec)`、`.download()`、`.fill(T v)`、`.guards_intact(R) -> bool`（失败时写入 R）。
- `hk::check_close(R, got, ref, atol, rtol, what, cols=0) -> bool`（`ref` 为 `std::vector<double>`）；`hk::check_equal(R, got, ref, what, cols=0) -> bool`（整型）。
- `hk::fail(R, fmt, ...)`：设 `R.passed=false` 并追加 printf 风格消息。
- `hk::time_best_ms(launch_fn, warmup=3, iters=20) -> double`；`hk::copy_bandwidth_gbs(bytes) -> double`（读+写字节 / 时间）。
- `R.metric`、`R.unit`：性能用例写入实测值（如 `"GB/s"`）。
- 进程协议：`--list` 输出 JSON 数组 `[{"name":..,"points":..,"perf":..}]`；`--case NAME` 输出一行 `@@RESULT {"name":..,"passed":..,"msg":..,"metric":..,"unit":..}`；无参数时在进程内顺序运行全部用例并打印人类可读结果。

- [ ] **Step 1: 建立 `.gitignore` 与 `CLAUDE.md`**

`.gitignore`:
```
build/
*.o
__pycache__/
.verify-tmp/
```

`CLAUDE.md`（完整内容）:
```markdown
# HIP / CK-Tile 学习仓库 —— 项目约定

## 定位
- 学习者：了解 GPU 概念、未系统写过 kernel；会 C++ 但模板元编程不熟；目标是在 MI308X 上做 mega kernel 研究。
- 一律用中文交流。用户常用语音输入："Hive"=HIP，"Cute/Catalyst/CkTile"≈CK-Tile，不确定时确认。
- 第一优先级：学习连续性——每个 Assignment 只用此前 Tutorial 讲过的概念。

## 结构
- 每个 Lab 一个文件夹 `labN-<name>/`，内含 `tutorial.html`、`Makefile`、`answers.md`、`aK-<name>/{<name>.hip,test.cpp,README.md}`。
- 参考答案与 mutants：`solutions/labN-<name>/aK-<name>/{<name>.hip,mutants/*.hip}`。mutants 绝不放进学生目录。
- 设计文档：`docs/superpowers/specs/`；计划：`docs/superpowers/plans/`；审查报告：`docs/reviews/`。

## 挖空规范
- 难度 1 填函数体 / 难度 2 补全设计 / 难度 3 自主设计，随课程推进过渡。
- 格式：`// TODO(aK-n): <中文说明>（讲义 §x.y）` + `// >>> 你的代码 <<<`。
- README：任务、分级 Hints（Hint 1 方向、Hint 2 关键步骤，不给代码）、书面问题。
- 骨架必须能编译，且原样提交时至少一个用例 FAIL。

## 判分规范
- test.cpp 通过 `#include STUDENT_SRC` 引入学生文件，只依赖 `common/harness.hpp`。
- 正确性：多尺寸（含 1、非整除、较大）、多种子、容差按类型、GuardedBuffer 检测越界、输入不被修改。
- 性能：计入分数；阈值校准（参考答案与慢 mutant 各留 ≥1.5× 余量）；内存类以同进程 D2D 拷贝带宽为基准。
- `make verify` 全绿（solutions 全过、每个 mutant 至少挂一项、骨架不满分）才算 Assignment 完成。
- mutant 必须确定性失败，不依赖竞态。

## 环境
- ROCm 7.2 `/opt/rocm`，8× MI308X gfx942（80 CU、wave64、LDS 64 KiB）。CK-Tile：`/opt/rocm/include/ck_tile`。无 numpy/torch。
- 编译：`/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942`。
- 机器多人共享：grade 自动选空闲 GPU，用例 60 s 超时，单次显存 ≤ 2 GB。

## CK-Tile 已知坑（实测）
- `tile_distribution_encoding` 只有 1 个 P 维时，P 只对应 lane id（0–63），整个 block 只有一个 wavefront 生效。block 级分布要用 2 个 P 维（warp id, lane id）。
- `make_tile_window` 的原点用 `{static_cast<index_t>(blockIdx.x) * Tile}`，否则 unsigned→int 窄化报错。

## Tutorial 风格
- 共用 `common/tutorial.css` / `common/tutorial.js`；左侧目录、代码高亮、SVG 图示、交互小部件、浅色/深色模式；每节末尾标注对应 Assignment。
- 外部脚本仅 cdnjs 且固定版本。
```

- [ ] **Step 2: 修正 spec §3**

把 spec 中 `│   │   └── mutants/          # 故意写错的变体，verify 时必须 FAIL` 一行删除，并把 `solutions/` 一行改为：
```
└── solutions/labN-<name>/aK-<name>/{<name>.hip, mutants/*.hip}   # 参考答案与 mutants（mutants 不进学生目录，避免泄露答案）
```

- [ ] **Step 3: 写 harness 自测（先写测试）**

`common/harness_selftest.cpp`:
```cpp
#include "harness.hpp"

__global__ void write_n_plus_one(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) p[i] = 1.0f;  // 故意越界写一个元素
}
__global__ void write_n(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f;
}

TEST_CASE(guard_detects_overflow, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_n_plus_one<<<4, 256>>>(b.get(), 1000);
  HIP_CHECK(hipDeviceSynchronize());
  hk::Result tmp;
  if (b.guards_intact(tmp)) hk::fail(R, "越界写未被检测到");
}

TEST_CASE(guard_accepts_inbounds, 1) {
  hk::GuardedBuffer<float> b(1000);
  write_n<<<4, 256>>>(b.get(), 1000);
  HIP_CHECK(hipDeviceSynchronize());
  b.guards_intact(R);
  auto h = b.download();
  for (float v : h) if (v != 1.0f) { hk::fail(R, "数据区错误"); break; }
}

TEST_CASE(guard_patterns_differ, 1) {
  hk::GuardedBuffer<float> a(16), b(16);
  if (a.pattern == b.pattern) hk::fail(R, "相邻两个缓冲区的哨兵模式相同");
}

TEST_CASE(check_close_behaviour, 1) {
  hk::Result a, b, c;
  std::vector<double> ref = {1.0, 2.0, 3.0};
  if (!hk::check_close(a, std::vector<float>{1.0f, 2.0f, 3.0f}, ref, 1e-6, 1e-6, "ok")) hk::fail(R, "相等误判");
  if (hk::check_close(b, std::vector<float>{1.0f, 2.5f, 3.0f}, ref, 1e-6, 1e-6, "bad")) hk::fail(R, "不等未判出");
  if (b.msg.find("[1]") == std::string::npos) hk::fail(R, "失败信息未包含下标: %s", b.msg.c_str());
  if (hk::check_close(c, std::vector<float>{1.0f, NAN, 3.0f}, ref, 1e9, 1e9, "nan")) hk::fail(R, "NaN 未判出");
}

TEST_CASE(always_fails_for_protocol_check, 0) {
  hk::fail(R, "预期失败: 含 \"引号\" 与\n换行");
}

PERF_CASE(copy_bandwidth_positive, 0) {
  R.metric = hk::copy_bandwidth_gbs(size_t(256) << 20);
  R.unit = "GB/s";
  if (!(R.metric > 100.0)) hk::fail(R, "拷贝带宽异常: %.1f", R.metric);
}

HK_MAIN()
```

- [ ] **Step 4: 编译自测，确认失败（harness.hpp 尚不存在）**

Run: `/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -Icommon common/harness_selftest.cpp -o build/selftest`
Expected: 编译错误 `'harness.hpp' file not found`

- [ ] **Step 5: 实现 `common/harness.hpp`**

```cpp
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
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
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
  int randint(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
};

inline std::vector<float> random_floats(size_t n, uint64_t seed, float lo = -1.f, float hi = 1.f) {
  Rng r(seed);
  std::vector<float> v(n);
  for (auto& x : v) x = r.uniform(lo, hi);
  return v;
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
    static const unsigned char kPatterns[] = {0xA5, 0x5A, 0xC3, 0x3C, 0x96, 0x69};
    static int counter = 0;
    pattern = kPatterns[counter++ % 6];
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
    bool ok = std::isfinite(g) && std::fabs(g - e) <= atol + rtol * std::fabs(e);
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
        std::fflush(stdout);
        std::printf("\n@@RESULT {\"name\":\"%s\",\"passed\":%s,\"msg\":\"%s\",\"metric\":%.6g,\"unit\":\"%s\"}\n",
                    c.name.c_str(), R.passed ? "true" : "false", json_escape(R.msg).c_str(), R.metric,
                    json_escape(R.unit).c_str());
        return 0;
      }
    std::fprintf(stderr, "未知用例: %s\n", argv[2]);
    return 3;
  }
  int pass = 0;
  for (auto& c : cases) {
    if (argc >= 2 && c.name != argv[1]) continue;
    Result R;
    run_one(c, R);
    pass += R.passed;
    std::printf("%-40s %s", c.name.c_str(), R.passed ? "OK" : "FAIL");
    if (c.perf) std::printf("  (%.1f %s)", R.metric, R.unit.c_str());
    std::printf("\n");
    if (!R.msg.empty()) std::printf("    %s\n", R.msg.c_str());
  }
  return 0;
}

}  // namespace hk

#define HK_MAIN() \
  int main(int argc, char** argv) { return hk::run_main(argc, argv); }
```

- [ ] **Step 6: 编译并运行自测**

Run:
```bash
mkdir -p build && /opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -Icommon common/harness_selftest.cpp -o build/selftest
./build/selftest --list
for c in guard_detects_overflow guard_accepts_inbounds guard_patterns_differ check_close_behaviour copy_bandwidth_positive always_fails_for_protocol_check; do ./build/selftest --case $c | grep @@RESULT; done
```
Expected: `--list` 输出 6 个用例的 JSON；除 `always_fails_for_protocol_check` 外 `"passed":true`；最后一个 `"passed":false` 且 msg 中引号被转义为 `\"`、换行为 `\n`；`python3 -c 'import json,sys; [json.loads(l[9:]) for l in sys.stdin]'` 能解析所有 `@@RESULT` 行。

- [ ] **Step 7: Commit**

```bash
git add .gitignore CLAUDE.md common/harness.hpp common/harness_selftest.cpp docs/superpowers/specs
git commit -m "feat(common): add test harness and project conventions"
```

---
### Task 2: 判分驱动 `grade.py`、校验 `verify.py`、公共 `lab.mk`

**Files:**
- Create: `common/grade.py`, `common/verify.py`, `common/lab.mk`, `common/test_grade.py`
- Create fixtures: `common/fixtures/fake-lab/a1-fake/fake.hip`, `common/fixtures/fake-lab/a1-fake/test.cpp`, `common/fixtures/solutions/fake-lab/a1-fake/fake.hip`, `common/fixtures/solutions/fake-lab/a1-fake/mutants/m1_wrong.hip`, `common/fixtures/solutions/fake-lab/a1-fake/mutants/m2_also_wrong.hip`

**Interfaces:**
- Consumes: Task 1 进程协议（`--list` / `--case` / `@@RESULT`）。
- Produces（verify.py 与各 Lab Makefile 使用）:
  - `grade.discover(lab_dir) -> list[Path]`：按名字排序的 `aK-*` 子目录（需含 `test.cpp`）。
  - `grade.student_src(assign_dir) -> Path`：目录内唯一的 `*.hip`。
  - `grade.build(assign_dir, src, out_bin) -> (bool, str)`
  - `grade.grade_assignment(assign_dir, src=None, quiet=False) -> dict`：`{"name", "built", "build_log", "cases":[{"name","points","perf","passed","msg","metric","unit"}], "score", "max"}`
  - `grade.select_gpu() -> None`：未设置 `HIP_VISIBLE_DEVICES` 时选最空闲卡写入 `os.environ`。
  - CLI：`python3 common/grade.py <lab_dir> [--only aK]`
  - CLI：`python3 common/verify.py <lab_dir> [--solutions-root DIR]`，有问题时退出码 1。
  - 环境变量 `HK_TIMEOUT`（秒，默认 60）。
  - `lab.mk` 目标：`make grade`、`make grade A=a2`、`make run A=a2 [CASE=name]`、`make verify`。

- [ ] **Step 1: 写 fixtures**

`common/fixtures/fake-lab/a1-fake/fake.hip`（学生骨架：返回错误值）:
```cpp
int student_value() {
  // TODO(a1-1): 返回 42
  // >>> 你的代码 <<<
  return 0;
}
```

`common/fixtures/solutions/fake-lab/a1-fake/fake.hip`:
```cpp
int student_value() { return 42; }
```

`common/fixtures/solutions/fake-lab/a1-fake/mutants/m1_wrong.hip`:
```cpp
int student_value() { return 41; }
```

`common/fixtures/solutions/fake-lab/a1-fake/mutants/m2_also_wrong.hip`:
```cpp
int student_value() { return -42; }
```

`common/fixtures/fake-lab/a1-fake/test.cpp`（用环境变量 `FAKE_MODE` 控制特殊行为，便于测试 grade 的鲁棒性；正常模式下只有 `value_is_42` 生效）:
```cpp
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
```

- [ ] **Step 2: 写失败的 unittest `common/test_grade.py`**

```python
import os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade  # noqa: E402

FIX = HERE / "fixtures"
LAB = FIX / "fake-lab"
SOL = FIX / "solutions"


class GradeTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("FAKE_MODE", None)
        os.environ["HK_TIMEOUT"] = "5"

    def test_discover_and_src(self):
        a = grade.discover(LAB)
        self.assertEqual([p.name for p in a], ["a1-fake"])
        self.assertEqual(grade.student_src(a[0]).name, "fake.hip")

    def test_skeleton_scores_partial(self):
        r = grade.grade_assignment(LAB / "a1-fake", quiet=True)
        self.assertTrue(r["built"])
        self.assertEqual(r["max"], 6)
        self.assertEqual(r["score"], 3)  # value_is_42 失败，其余 3 个用例通过

    def test_solution_full_score(self):
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        self.assertEqual(r["score"], r["max"])

    def test_crash_isolated(self):
        os.environ["FAKE_MODE"] = "crash"
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_crash"]["passed"])
        self.assertIn("崩溃", by["maybe_crash"]["msg"])
        self.assertTrue(by["value_is_42"]["passed"])

    def test_hang_times_out(self):
        os.environ["FAKE_MODE"] = "hang"
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_hang"]["passed"])
        self.assertIn("超时", by["maybe_hang"]["msg"])
        self.assertTrue(by["value_is_42"]["passed"])

    def test_perf_retry(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["FAKE_MODE"] = "flaky:" + os.path.join(d, "marker")
            r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
            by = {c["name"]: c for c in r["cases"]}
            self.assertTrue(by["maybe_flaky_perf"]["passed"])
            self.assertEqual(by["maybe_flaky_perf"]["metric"], 2)

    def test_perf_failure_mentions_retries(self):
        os.environ["FAKE_MODE"] = "flaky:/nonexistent-dir/marker"  # 标记文件无法创建 → 每次都失败
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_flaky_perf"]["passed"])
        self.assertIn("已重试 3 次", by["maybe_flaky_perf"]["msg"])

    def test_build_failure_reported(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "fake.hip"
            bad.write_text("int student_value() { return }\n")
            r = grade.grade_assignment(LAB / "a1-fake", src=bad, quiet=True)
            self.assertFalse(r["built"])
            self.assertEqual(r["score"], 0)
            self.assertIn("error", r["build_log"])

    def test_verify_passes_on_fixture(self):
        p = subprocess.run([sys.executable, str(HERE / "verify.py"), str(LAB), "--solutions-root", str(SOL)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_verify_catches_weak_tests(self):
        # 把 mutant 换成与答案相同的实现：verify 必须报告"mutant 未被抓住"
        with tempfile.TemporaryDirectory() as d:
            sol = Path(d) / "solutions"
            shutil.copytree(SOL, sol)
            (sol / "fake-lab/a1-fake/mutants/m1_wrong.hip").write_text("int student_value() { return 42; }\n")
            p = subprocess.run([sys.executable, str(HERE / "verify.py"), str(LAB), "--solutions-root", str(sol)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn("m1_wrong", p.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python3 common/test_grade.py -v`
Expected: `ModuleNotFoundError: No module named 'grade'`

- [ ] **Step 4: 实现 `common/grade.py`**

```python
#!/usr/bin/env python3
"""判分驱动：编译每个 Assignment，逐个用例在独立进程中运行，汇总分数。仅用标准库。"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
REPO = COMMON.parent
HIPCC = "/opt/rocm/bin/hipcc"
CXXFLAGS = ["-std=c++20", "-O2", "--offload-arch=gfx942", f"-I{COMMON}"]
PERF_TRIES = 3


def timeout_s():
    return float(os.environ.get("HK_TIMEOUT", "60"))


def discover(lab_dir):
    lab_dir = Path(lab_dir)
    return sorted(p for p in lab_dir.iterdir()
                  if p.is_dir() and re.match(r"a\d+-", p.name) and (p / "test.cpp").exists())


def student_src(assign_dir):
    hips = sorted(Path(assign_dir).glob("*.hip"))
    if len(hips) != 1:
        raise RuntimeError(f"{assign_dir} 中应恰好有一个 .hip 文件，实际 {len(hips)} 个")
    return hips[0]


def build(assign_dir, src, out_bin):
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [HIPCC, *CXXFLAGS, f'-DSTUDENT_SRC="{Path(src).resolve()}"',
           str(Path(assign_dir).resolve() / "test.cpp"), "-o", str(out_bin)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0, p.stdout + p.stderr


def select_gpu():
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showuse", "--showmemuse", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out[out.index("{"):])
    except Exception:
        return
    best = None
    for card, info in data.items():
        m = re.match(r"card(\d+)$", card)
        if not m:
            continue
        try:
            load = float(info.get("GPU use (%)", 100)) + float(info.get("GPU Memory Allocated (VRAM%)", 100))
        except ValueError:
            continue
        if best is None or load < best[0]:
            best = (load, m.group(1))
    if best:
        os.environ["HIP_VISIBLE_DEVICES"] = best[1]


def gpu_load_note():
    """性能用例失败时附上当前所用 GPU 的占用，帮助区分"代码慢"与"别人在用卡"。"""
    dev = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi", "-d", dev, "--showuse", "--showmemuse", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        info = next(iter(json.loads(out[out.index("{"):]).values()))
        return f"（当前 GPU {dev} 占用：计算 {info.get('GPU use (%)')}%，显存 {info.get('GPU Memory Allocated (VRAM%)')}%）"
    except Exception:
        return ""


def run_case(binary, name):
    try:
        p = subprocess.run([str(binary), "--case", name], capture_output=True, text=True, timeout=timeout_s())
    except subprocess.TimeoutExpired:
        return {"passed": False, "msg": f"超时（>{timeout_s():.0f} s）：可能死循环、死锁或越界导致挂起", "metric": 0, "unit": ""}
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT "):
            r = json.loads(line[len("@@RESULT "):])
            return {k: r[k] for k in ("passed", "msg", "metric", "unit")}
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
    return {"passed": False, "msg": f"进程崩溃（退出码 {p.returncode}）\n{tail}", "metric": 0, "unit": ""}


def grade_assignment(assign_dir, src=None, quiet=False):
    assign_dir = Path(assign_dir)
    src = Path(src) if src else student_src(assign_dir)
    binary = assign_dir.parent / "build" / assign_dir.name / f"test-{os.getpid()}-{abs(hash(str(src))) % 10**8}"
    res = {"name": assign_dir.name, "built": False, "build_log": "", "cases": [], "score": 0, "max": 0}
    ok, log = build(assign_dir, src, binary)
    res["built"], res["build_log"] = ok, log
    if not ok:
        res["max"] = sum(c["points"] for c in list_cases_from_source(assign_dir))
        if not quiet:
            print(f"{assign_dir.name}: 编译失败（0 分）\n{log}")
        return res
    cases = json.loads(subprocess.run([str(binary), "--list"], capture_output=True, text=True,
                                      timeout=timeout_s()).stdout)
    if not quiet:
        print(f"{assign_dir.name}")
    for c in cases:
        r = run_case(binary, c["name"])
        if c["perf"] and not r["passed"]:
            for _ in range(PERF_TRIES - 1):
                r2 = run_case(binary, c["name"])
                if r2["passed"] or r2["metric"] > r["metric"]:
                    r = r2
                if r["passed"]:
                    break
            if not r["passed"]:
                r["msg"] += f"\n已重试 {PERF_TRIES} 次{gpu_load_note()}"
        case = {**c, **r}
        res["cases"].append(case)
        res["max"] += c["points"]
        res["score"] += c["points"] if r["passed"] else 0
        if not quiet:
            perf = f"  [{r['metric']:.1f} {r['unit']}]" if c["perf"] else ""
            status = "OK" if r["passed"] else "FAIL"
            print(f"  {c['name']:.<44} {status} ({c['points'] if r['passed'] else 0}/{c['points']}){perf}")
            if not r["passed"] and r["msg"]:
                print("    " + r["msg"].replace("\n", "\n    "))
    try:
        binary.unlink()
    except OSError:
        pass
    return res


def list_cases_from_source(assign_dir):
    """编译失败时从 test.cpp 源码估算满分，便于显示 0/满分。"""
    text = (Path(assign_dir) / "test.cpp").read_text()
    return [{"name": m.group(2), "points": int(m.group(3))}
            for m in re.finditer(r"\b(TEST_CASE|PERF_CASE)\((\w+),\s*(\d+)\)", text)]


def main():
    ap = argparse.ArgumentParser(description="判分")
    ap.add_argument("lab_dir")
    ap.add_argument("--only", help="只判某个 Assignment，例如 a2")
    args = ap.parse_args()
    select_gpu()
    print(f"使用 GPU: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '(默认)')}")
    total = mx = 0
    for a in discover(args.lab_dir):
        if args.only and not a.name.startswith(args.only + "-"):
            continue
        r = grade_assignment(a)
        total += r["score"]
        mx += r["max"]
    print(f"Score: {total}/{mx}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 实现 `common/verify.py`**

```python
#!/usr/bin/env python3
"""校验判分器：参考答案必须满分；每个 mutant 至少挂一项；学生骨架必须能编译且不满分。"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade  # noqa: E402

MIN_MUTANTS = 2


def verify_lab(lab_dir, sol_root):
    lab_dir = Path(lab_dir).resolve()
    problems = []
    for a in grade.discover(lab_dir):
        sdir = Path(sol_root) / lab_dir.name / a.name
        src_name = grade.student_src(a).name
        print(f"== {a.name}")
        sol = grade.grade_assignment(a, src=sdir / src_name, quiet=True)
        if not sol["built"]:
            problems.append(f"{a.name}: 参考答案编译失败\n{sol['build_log']}")
        elif sol["score"] != sol["max"]:
            bad = [f"{c['name']}: {c['msg']}" for c in sol["cases"] if not c["passed"]]
            problems.append(f"{a.name}: 参考答案未满分 {sol['score']}/{sol['max']}\n  " + "\n  ".join(bad))
        else:
            perf = [f"{c['name']}={c['metric']:.1f}{c['unit']}" for c in sol["cases"] if c["perf"]]
            print(f"   solution    满分 {sol['score']}/{sol['max']} {' '.join(perf)}")
        mutants = sorted((sdir / "mutants").glob("*.hip"))
        if len(mutants) < MIN_MUTANTS:
            problems.append(f"{a.name}: mutant 数量 {len(mutants)} < {MIN_MUTANTS}")
        for m in mutants:
            r = grade.grade_assignment(a, src=m, quiet=True)
            if not r["built"]:
                problems.append(f"{a.name}: mutant {m.name} 编译失败（mutant 必须能编译）")
                continue
            failed = [c["name"] for c in r["cases"] if not c["passed"]]
            if not failed:
                problems.append(f"{a.name}: mutant {m.name} 未被任何用例抓住")
            else:
                print(f"   mutant      {m.name:<32} 被抓住: {', '.join(failed)}")
        sk = grade.grade_assignment(a, quiet=True)
        if not sk["built"]:
            problems.append(f"{a.name}: 学生骨架无法编译\n{sk['build_log']}")
        elif sk["score"] >= sk["max"]:
            problems.append(f"{a.name}: 学生骨架原样即满分")
        else:
            print(f"   skeleton    {sk['score']}/{sk['max']}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lab_dir")
    ap.add_argument("--solutions-root", default=str(grade.REPO / "solutions"))
    args = ap.parse_args()
    grade.select_gpu()
    problems = verify_lab(args.lab_dir, args.solutions_root)
    if problems:
        print("\nVERIFY FAILED:")
        for p in problems:
            print(" - " + p)
        sys.exit(1)
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 实现 `common/lab.mk`**

```make
# 各 Lab 的 Makefile 只需写：include ../common/lab.mk
COMMON := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LAB    := $(abspath .)
A      ?=
CASE   ?=

.PHONY: grade run verify clean
grade:
	@python3 $(COMMON)/grade.py $(LAB) $(if $(A),--only $(A))

# make run A=a2 [CASE=name]：编译并直接运行（人类可读输出，便于调试/打印）
run:
	@test -n "$(A)" || (echo "用法: make run A=a2 [CASE=用例名]"; exit 1)
	@d=$$(ls -d $(LAB)/$(A)-*/); src=$$(ls $$d*.hip); mkdir -p $(LAB)/build; \
	/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -I$(COMMON) -DSTUDENT_SRC="\"$$src\"" \
	  $$d/test.cpp -o $(LAB)/build/run-$(A) && $(LAB)/build/run-$(A) $(CASE)

verify:
	@python3 $(COMMON)/verify.py $(LAB)

clean:
	rm -rf $(LAB)/build
```

- [ ] **Step 7: 运行 unittest，确认通过**

Run: `python3 common/test_grade.py -v`
Expected: 10 个测试全部 `ok`（hang 测试约 5 s）。

- [ ] **Step 8: Commit**

```bash
git add common/grade.py common/verify.py common/lab.mk common/test_grade.py common/fixtures
git commit -m "feat(common): add grading driver, verifier and lab makefile"
```

---
### Task 3: Tutorial 设计系统（CSS/JS/模板/总目录）

**Files:**
- Create: `common/tutorial.css`, `common/tutorial.js`, `common/tutorial-template.html`, `site/index.html`

**Interfaces（各 Lab 的 tutorial.html 依赖）:**
- 页面骨架：`<body class="tut">` → `<nav class="toc">`（由 JS 根据 `h2[id]`/`h3[id]` 自动生成）+ `<main class="content">`。
- 组件 class：`.callout.note` / `.callout.warn` / `.callout.pitfall`（常见坑）；`.assignment-card`（含 `data-assign="a2"`，显示"现在去做 Assignment a2"、文件路径与 `make grade A=a2` 命令）；`figure.diagram`（内联 SVG，颜色只用 CSS 变量）；`.widget[data-widget="<name>"]`（交互小部件挂载点，由页面自身脚本 `window.TUT.registerWidget(name, initFn)` 注册）；`.kbd`、`table.facts`（硬件参数表）。
- 代码高亮：highlight.js 11.9.0（cdnjs，固定版本），语言 `cpp`、`bash`；`<pre><code class="language-cpp">`。
- 主题：`:root` 定义颜色 token；`@media (prefers-color-scheme: dark)` + `:root[data-theme=dark|light]` 覆盖；右上角主题切换按钮（`localStorage` 记忆，try/catch 包裹）。
- `tutorial.js` 导出 `window.TUT = { registerWidget(name, fn) }`，DOMContentLoaded 时：生成 TOC、滚动高亮当前节、初始化 hljs、调用已注册 widget 的 `fn(el)`。

- [ ] **Step 1: 调用 frontend-design skill 确定视觉方向**

要求：技术讲义气质（长时间阅读舒适、中文排版优先，正文字体 `"Noto Sans SC", system-ui`，代码字体 `"JetBrains Mono", ui-monospace`，通过 Google Fonts 加载），区别于模板化文档站；重点设计代码块、callout、SVG 图示配色与 assignment-card。把方向写成 `common/tutorial.css` 顶部注释（5 行以内）。

- [ ] **Step 2: 实现 `common/tutorial.css`、`common/tutorial.js`、`common/tutorial-template.html`**

模板页包含每种组件各一个示例（含一个 demo widget：滑块改变数字并重画一个 SVG 方格），作为后续 Lab 讲义的复制起点。

- [ ] **Step 3: 实现 `site/index.html`**

课程总目录：课程简介、学习方式（读讲义 → 做作业 → `make grade`）、Lab 0–8 卡片（Lab 0–3 链接到 `../labN-*/tutorial.html`，Lab 4–8 标"即将推出"并写一句主题），共用同一套 CSS。

- [ ] **Step 4: 浏览器验证**

用 Playwright 打开 `file://<repo>/common/tutorial-template.html` 与 `site/index.html`：
- `browser_console_messages(level="error")` 为空；
- 截图浅色与深色（`browser_emulate_media colorScheme`）各一张，人工检查对比度；
- `browser_resize(390, 844)` 后执行 `document.documentElement.scrollWidth <= window.innerWidth`，Expected: `true`；
- demo widget 拖动滑块后 SVG 方格数量变化（`browser_evaluate` 读取 `rect` 数量）。

- [ ] **Step 5: Commit**

```bash
git add common/tutorial.css common/tutorial.js common/tutorial-template.html site/index.html
git commit -m "feat(site): add tutorial design system and course index"
```

---

### Task 4: Lab 0 —— 环境与工具

**目标（学习者视角）：** 学会"编辑 → `make grade` → 读失败信息 → 改正"的循环；写出第一个 kernel；用 HIP API 查询硬件参数，建立对 MI308X 的数字直觉；了解 `rocprofv3`。

**Files:**
- Create: `lab0-env/Makefile`, `lab0-env/tutorial.html`, `lab0-env/answers.md`
- Create: `lab0-env/a1-hello/{hello.hip,test.cpp,README.md}`, `lab0-env/a2-device-query/{device_query.hip,test.cpp,README.md}`
- Create: `solutions/lab0-env/a1-hello/hello.hip` + `mutants/{m1_no_bounds.hip,m2_thread_only.hip}`
- Create: `solutions/lab0-env/a2-device-query/device_query.hip` + `mutants/{m1_hardcoded_cu.hip,m2_lds_wrong_attr.hip}`

**Interfaces:**
- a1：`void launch_fill_ids(int* out, int n, hipStream_t stream);` —— `out[i] = i`（0 ≤ i < n）。
- a2：
```cpp
struct DeviceSummary {
  char arch[64];          // gcnArchName 中冒号前部分，如 "gfx942"
  int  num_cu;            // hipDeviceAttributeMultiprocessorCount
  int  warp_size;         // hipDeviceAttributeWarpSize
  int  max_threads_per_block;
  int  lds_bytes_per_block;      // hipDeviceAttributeMaxSharedMemoryPerBlock
  long long global_mem_bytes;    // hipDeviceProp_t::totalGlobalMem
  int  clock_mhz;                // hipDeviceAttributeClockRate / 1000
};
DeviceSummary query_device(int device);
```

- [ ] **Step 1: `lab0-env/Makefile`**

```make
include ../common/lab.mk
```

- [ ] **Step 2: a1 测试 `lab0-env/a1-hello/test.cpp`**

```cpp
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
```

- [ ] **Step 3: a1 参考答案 `solutions/lab0-env/a1-hello/hello.hip`**

```cpp
#include <hip/hip_runtime.h>

__global__ void fill_ids_kernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

void launch_fill_ids(int* out, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  fill_ids_kernel<<<grid, block, 0, stream>>>(out, n);
}
```

- [ ] **Step 4: a1 学生骨架 `lab0-env/a1-hello/hello.hip`**

与参考答案相同的结构，但：kernel 体为
```cpp
  // TODO(a1-1): 计算本线程负责的全局下标 i，并在 i < n 时写 out[i] = i（讲义 §3.2）
  //   提示：全局下标 = blockIdx.x * blockDim.x + threadIdx.x
  // >>> 你的代码 <<<
```
launch 函数中 `const int grid = 1;` 并标注
```cpp
  // TODO(a1-2): grid 要足够大以覆盖全部 n 个元素（向上取整），见讲义 §3.3
  // >>> 你的代码 <<<
```
（骨架可编译；原样提交时全部 FAIL。）

- [ ] **Step 5: a1 mutants**

`m1_no_bounds.hip`：参考答案去掉 `if (i < n)`（直接 `out[i] = i;`）→ `not_multiple_of_block` 因哨兵 FAIL。
`m2_thread_only.hip`：`int i = threadIdx.x;` → `one_block_exact` 之外的多 block 用例 FAIL。

- [ ] **Step 6: a2 测试 `lab0-env/a2-device-query/test.cpp`**

```cpp
#include "harness.hpp"
#include STUDENT_SRC
#include <string>

static DeviceSummary expected(int dev) {
  DeviceSummary e{};
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, dev));
  std::string a = p.gcnArchName;
  a = a.substr(0, a.find(':'));
  std::snprintf(e.arch, sizeof(e.arch), "%s", a.c_str());
  HIP_CHECK(hipDeviceGetAttribute(&e.num_cu, hipDeviceAttributeMultiprocessorCount, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.warp_size, hipDeviceAttributeWarpSize, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, dev));
  HIP_CHECK(hipDeviceGetAttribute(&e.lds_bytes_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
  e.global_mem_bytes = (long long)p.totalGlobalMem;
  int khz;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, dev));
  e.clock_mhz = khz / 1000;
  return e;
}

#define EXPECT_FIELD(f) if (g.f != e.f) hk::fail(R, #f ": 期望 %lld，得到 %lld", (long long)e.f, (long long)g.f)

TEST_CASE(arch_name, 1) {
  auto g = query_device(0), e = expected(0);
  if (std::string(g.arch) != e.arch) hk::fail(R, "arch: 期望 \"%s\"，得到 \"%s\"", e.arch, g.arch);
}
TEST_CASE(compute_units_and_wave, 2) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(num_cu); EXPECT_FIELD(warp_size); }
TEST_CASE(block_limits, 2) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(max_threads_per_block); EXPECT_FIELD(lds_bytes_per_block); }
TEST_CASE(memory_and_clock, 1) { auto g = query_device(0), e = expected(0); EXPECT_FIELD(global_mem_bytes); EXPECT_FIELD(clock_mhz); }
HK_MAIN()
```

- [ ] **Step 7: a2 参考答案、骨架、mutants**

参考答案：按 Interfaces 注释逐字段调用对应 API（与 test 的 `expected` 等价，但学生需自己查 API；`arch` 用 `strncpy` 截掉冒号后缀）。
骨架：`DeviceSummary s{}; return s;`，每个字段一行 `// TODO(a2-n)`，注释写出应查询的属性名（难度 1，目的在于熟悉 API 文档和 `hipDeviceProp_t`）。
`m1_hardcoded_cu.hip`：`num_cu = 304`（MI300X 的值，不是本机 80）→ `compute_units_and_wave` FAIL。
`m2_lds_wrong_attr.hip`：`lds_bytes_per_block` 使用 `hipDeviceAttributeMaxRegistersPerBlock` → `block_limits` FAIL。

- [ ] **Step 8: 两个 README**

`a1-hello/README.md` 结构（后续所有 Assignment README 同此结构）：
```markdown
# Lab 0 · a1 hello：第一个 kernel
## 你要做什么
## 怎么判分（make grade A=a1；每个用例检查什么）
## Hints
<details><summary>Hint 1（方向）</summary>…</details>
<details><summary>Hint 2（关键步骤）</summary>…</details>
## 书面问题（写到 ../answers.md）
1. 若 n = 1000、blockDim = 256，启动了多少个线程？多出来的线程在做什么？
2. 去掉 `if (i < n)` 后，n = 1000 时会发生什么？为什么"结果看起来对"不代表程序正确？（提示：哨兵）
## 动手探索（不判分）
- 用 `rocprofv3 --kernel-trace -- ./build/run-a1` 看 kernel 名字与耗时。
```
a2 README 书面问题：本机 CU 数、每 CU 理论上同时驻留多少 wavefront（查 CDNA3 白皮书：每 SIMD 8 个 wave，每 CU 4 个 SIMD）；`max_threads_per_block=1024` 对应几个 wavefront。

- [ ] **Step 9: Lab 0 讲义 `lab0-env/tutorial.html`（从模板复制）**

章节：
1. 本课程怎么学（讲义 → 作业 → make grade → answers.md；如何读 FAIL 信息；`make run A=..` 调试）
2. 一台 MI308X 长什么样（GPU → XCD → CU → SIMD；SVG 图；`table.facts` 填本机实测：80 CU、wave64、LDS 64 KiB、HBM 容量、频率）
3. 第一个 kernel：`__global__`、`<<<grid, block>>>`、`blockIdx/blockDim/threadIdx`、全局下标；3.2 节下标计算 widget（输入 blockDim、n，显示每个线程映射到哪个元素，越界线程标灰）；3.3 grid 向上取整；末尾 assignment-card a1
4. 错误检查：`hipError_t`、异步 kernel 与 `hipDeviceSynchronize`、`HIP_CHECK`
5. 查询设备属性：`hipGetDeviceProperties` / `hipDeviceGetAttribute`；末尾 assignment-card a2
6. 工具：`rocm-smi`（看谁在用卡）、`HIP_VISIBLE_DEVICES`、`rocprofv3 --kernel-trace`

- [ ] **Step 10: `lab0-env/answers.md`**：列出两个 README 的书面问题编号与空白作答区。

- [ ] **Step 11: 验证**

Run: `cd lab0-env && make verify`
Expected: `VERIFY OK`；solution 满分；4 个 mutant 均被抓住；骨架 0/6 与 0/6。
Run: `make grade`（骨架）Expected: 输出中文 FAIL 信息、`Score: 0/12`。

- [ ] **Step 12: Commit**

```bash
git add lab0-env solutions/lab0-env
git commit -m "feat(lab0): environment lab with hello kernel and device query"
```

---
### Task 5: Lab 1 · a1 vector add 与 grid-stride loop

**Files:**
- Create: `lab1-hip-basics/Makefile`（内容 `include ../common/lab.mk`）
- Create: `lab1-hip-basics/a1-vector-add/{vector_add.hip,test.cpp,README.md}`
- Create: `solutions/lab1-hip-basics/a1-vector-add/vector_add.hip` + `mutants/{m1_no_bounds.hip,m2_no_stride.hip,m3_alpha_ignored.hip}`

**Interfaces:**
```cpp
// Part A：c[i] = a[i] + b[i]，grid 由学生决定
void launch_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream);
// Part B：y[i] = alpha * x[i] + y[i]。grid 固定为 kSaxpyGrid 个 block × 256 线程（骨架中给定不可改），
// 因此必须用 grid-stride loop 覆盖任意 n。
constexpr int kSaxpyGrid = 160;
void launch_saxpy(float alpha, const float* x, float* y, int n, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab1-hip-basics/a1-vector-add/test.cpp`**

```cpp
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
```

（`check_close` atol=rtol=0 用于加法：GPU 与 CPU 的 float 加法逐位一致。saxpy 用 `fmaf` 作参考并给 1e-6 容差，允许学生写成 `alpha*x+y` 或 fma。）

- [ ] **Step 2: 参考答案 `solutions/lab1-hip-basics/a1-vector-add/vector_add.hip`**

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

void launch_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  vector_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
}

constexpr int kSaxpyGrid = 160;

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = alpha * x[i] + y[i];
}

void launch_saxpy(float alpha, const float* x, float* y, int n, hipStream_t stream) {
  saxpy_kernel<<<kSaxpyGrid, 256, 0, stream>>>(alpha, x, y, n);
}
```

- [ ] **Step 3: 学生骨架 `lab1-hip-basics/a1-vector-add/vector_add.hip`**

保留全部函数签名、`kSaxpyGrid` 与 `saxpy_kernel<<<kSaxpyGrid, 256, 0, stream>>>` 一行（注释"不要修改 grid"），挖空：
- `// TODO(a1-1): vector_add_kernel 体：计算全局下标并做边界检查（讲义 §2.3）`
- `// TODO(a1-2): launch_vector_add：选择 block 大小并计算 grid（讲义 §2.4）`，骨架中 `grid = 1, block = 64`
- `// TODO(a1-3): saxpy_kernel：grid 只有 kSaxpyGrid×256 个线程，用 grid-stride loop 覆盖所有 n（讲义 §2.5）`，骨架体为空

- [ ] **Step 4: mutants**

- `m1_no_bounds.hip`：vector_add 去掉 `if (i < n)` → `add_ragged` 哨兵 FAIL。
- `m2_no_stride.hip`：saxpy 改为 `int i = ...; if (i < n) y[i] = ...;`（无循环）→ `saxpy_needs_stride` FAIL。
- `m3_alpha_ignored.hip`：`y[i] = x[i] + y[i];` → saxpy 全部 FAIL。

- [ ] **Step 5: README**

任务分 Part A/B；Hint 1：为什么 grid 要向上取整；Hint 2：grid-stride 循环的步长 = 网格中线程总数。书面问题：
1. Part B 里 n = 1 时，有多少线程什么都没做？这样的浪费在什么情况下值得？
2. grid-stride loop 相比"线程数 = n"的写法有什么好处？（提示：launch 开销、复用线程、可调占用）
3. 用 `make run A=a1` 打印 `add_large` 的时间（`rocprofv3 --kernel-trace`），算出带宽 = 3 × 4 × n 字节 / 时间，与 MI308X HBM 峰值（讲义 §1 表格）比较。

- [ ] **Step 6: 验证**

Run: `cd lab1-hip-basics && make verify`（此时只有 a1）
Expected: `VERIFY OK`，3 个 mutant 都被抓住，骨架 < 满分。

- [ ] **Step 7: Commit**

```bash
git add lab1-hip-basics/Makefile lab1-hip-basics/a1-vector-add solutions/lab1-hip-basics/a1-vector-add
git commit -m "feat(lab1): a1 vector add and grid-stride saxpy"
```

---
### Task 6: Lab 1 · a2 矩阵转置（naive → LDS 分块）

**学习目标：** 2D 网格与 2D 下标；合并访存（coalescing）；LDS（`__shared__`）与 `__syncthreads()`；bank conflict 与 padding；用带宽衡量 kernel。第一次出现性能用例。

**Files:**
- Create: `lab1-hip-basics/a2-transpose/{transpose.hip,test.cpp,README.md}`
- Create: `solutions/lab1-hip-basics/a2-transpose/transpose.hip` + `mutants/{m1_wrong_leading_dim.hip,m2_lds_no_bounds.hip,m3_lds_is_naive.hip}`

**Interfaces:**
```cpp
// in: rows×cols 行主序；out: cols×rows 行主序，out[c*rows + r] = in[r*cols + c]
void launch_transpose_naive(const float* in, float* out, int rows, int cols, hipStream_t stream);
void launch_transpose_lds(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab1-hip-basics/a2-transpose/test.cpp`**

```cpp
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
// kLdsRatio 在 Step 6 校准后写死。
constexpr double kLdsRatio = 0.50;
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
```

- [ ] **Step 2: 参考答案 `solutions/lab1-hip-basics/a2-transpose/transpose.hip`**

```cpp
#include <hip/hip_runtime.h>

__global__ void transpose_naive_kernel(const float* in, float* out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[size_t(c) * rows + r] = in[size_t(r) * cols + c];
}

void launch_transpose_naive(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_naive_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

constexpr int TILE = 64;      // 每个 block 处理 TILE×TILE 的子块
constexpr int ROWS_PER = 4;   // block = TILE × ROWS_PER 线程，每线程处理 TILE/ROWS_PER 行

__global__ void transpose_lds_kernel(const float* in, float* out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];  // +1 padding 消除 bank conflict
  int c0 = blockIdx.x * TILE, r0 = blockIdx.y * TILE;
  // 1) 合并地读入：同一 wavefront 的线程读同一行的连续元素
  for (int dy = threadIdx.y; dy < TILE; dy += ROWS_PER) {
    int r = r0 + dy, c = c0 + threadIdx.x;
    if (r < rows && c < cols) tile[dy][threadIdx.x] = in[size_t(r) * cols + c];
  }
  __syncthreads();
  // 2) 合并地写出：输出的一行对应输入的一列，从 LDS 按列读取
  for (int dy = threadIdx.y; dy < TILE; dy += ROWS_PER) {
    int oc = r0 + threadIdx.x;  // 输出列 = 输入行
    int orow = c0 + dy;         // 输出行 = 输入列
    if (orow < cols && oc < rows) out[size_t(orow) * rows + oc] = tile[threadIdx.x][dy];
  }
}

void launch_transpose_lds(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 block(TILE, ROWS_PER);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transpose_lds_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}
```

- [ ] **Step 3: 学生骨架（难度 1 → 2）**

- naive 部分（难度 1）：给出 launch 函数完整内容，kernel 体留 `// TODO(a2-1): 由 2D 线程坐标得到 (r, c)，边界检查后写 out（讲义 §3.2）`。
- LDS 部分（难度 2）：只给 kernel 签名与 launch 签名，kernel 内写分步引导注释，**不给 TILE、block 形状、LDS 声明**：
```cpp
// TODO(a2-2): 设计你的分块转置（讲义 §3.4–3.6）
//   第 0 步：决定 tile 大小与 block 形状（提示：一个 wavefront 是 64 线程，让它读一整行连续元素）
//   第 1 步：声明 __shared__ 数组；想想为什么讲义建议多加 1 列
//   第 2 步：从 in 合并地读入 tile（注意矩阵边缘的越界）
//   第 3 步：同步——为什么必须在这里同步？
//   第 4 步：从 LDS 按"转置后的方向"读出，合并地写入 out
// >>> 你的代码 <<<
```
骨架中 `launch_transpose_lds` 暂时调用 naive 版本并标 `// TODO(a2-3): 换成你的 LDS kernel 与对应的 grid/block`（naive kernel 体为空，因此原样提交时所有用例 FAIL）。

- [ ] **Step 4: mutants**

- `m1_wrong_leading_dim.hip`：naive 写 `out[size_t(c) * cols + r]` → 非方阵 FAIL。
- `m2_lds_no_bounds.hip`：LDS kernel 去掉读入与写出的两处边界检查 → `lds_all_shapes` 在 (1,1)/(33,31) 处哨兵或数值 FAIL（输入、输出哨兵模式不同，保证确定性）。
- `m3_lds_is_naive.hip`：`launch_transpose_lds` 直接调用 naive kernel → `lds_bandwidth` FAIL（依赖 Step 6 校准）。

- [ ] **Step 5: 跑 verify 初查**

Run: `cd lab1-hip-basics && make verify`
Expected: solution 满分（含 `lds_bandwidth`）；m1、m2 被抓住；m3 是否被抓住取决于校准。

- [ ] **Step 6: 校准 `kLdsRatio`**

Run（GPU 空闲时，重复 5 次）:
```bash
for i in 1 2 3 4 5; do python3 -c "
import sys; sys.path.insert(0,'../common'); import grade
from pathlib import Path
a=Path('a2-transpose'); s=Path('../solutions/lab1-hip-basics/a2-transpose')
for src in [s/'transpose.hip', s/'mutants/m3_lds_is_naive.hip']:
    r=grade.grade_assignment(a, src=src, quiet=True)
    print(src.name, [ (c['metric'], c['msg'][-60:]) for c in r['cases'] if c['perf']])
"; done
```
从失败信息中读出本次拷贝带宽 copy，算出 solution 比例 S = 解的带宽/copy，naive 比例 N。要求 S/1.5 ≥ N×1.5，取 `kLdsRatio = sqrt(S×N)` 向下取两位小数，写回 test.cpp。若 S/N < 2.25，则：先尝试优化参考答案（每线程处理多元素、`float4` 读），仍不满足则删除 `lds_bandwidth`、删除 m3，并在 `docs/reviews/lab1.md` 记录原因。

- [ ] **Step 7: README**

Hint 1：画出 naive 版本中一个 wavefront 读/写的地址，哪个是连续的？Hint 2：LDS 作为"中转站"，读和写都能做到连续；LDS 有 32 个 bank，每个 4 字节宽，同一时刻访问同一 bank 的不同地址会串行化。书面问题：
1. naive 版本中读和写哪一个是非合并访存？为什么？
2. 去掉 `+1` padding 后结果还正确吗？性能会怎样？用 `make run A=a2 CASE=lds_bandwidth` 实测并解释。
3. 去掉 `__syncthreads()` 会出什么问题？为什么这种 bug 可能"偶尔对"？（这也是判分没有把它当作必挂 mutant 的原因。）

- [ ] **Step 8: 验证与 Commit**

Run: `make verify` → Expected: `VERIFY OK`。
```bash
git add lab1-hip-basics/a2-transpose solutions/lab1-hip-basics/a2-transpose
git commit -m "feat(lab1): a2 transpose naive and LDS tiled with bandwidth check"
```

---
### Task 7: Lab 1 · a3 归约（LDS 树形 → wavefront shuffle）

**学习目标：** 并行归约的两阶段结构（多 block 部分和 → 单 block 汇总）；LDS 树形归约与每层同步；wavefront 内 `__shfl_down`（**wave64**，不是 CUDA 的 32）；浮点求和误差；输出初始化。

**Files:**
- Create: `lab1-hip-basics/a3-reduction/{reduction.hip,test.cpp,README.md}`
- Create: `solutions/lab1-hip-basics/a3-reduction/reduction.hip` + `mutants/{m1_drop_tail.hip,m2_atomic_no_init.hip,m3_single_block.hip,m4_shfl_width32.hip}`

**Interfaces:**
```cpp
constexpr int kWorkspaceFloats = 4096;  // 测试提供的工作区大小（float 个数）
// *out = sum(in[0..n))。ws 是 kWorkspaceFloats 个 float 的设备工作区，内容未初始化。
void reduce_sum_lds (const float* in, int n, float* out, float* ws, hipStream_t stream);
void reduce_sum_shfl(const float* in, int n, float* out, float* ws, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab1-hip-basics/a3-reduction/test.cpp`**

```cpp
#include "harness.hpp"
#include STUDENT_SRC

using Fn = void (*)(const float*, int, float*, float*, hipStream_t);

// 返回 GPU 结果；out 与 ws 预先填 NaN，以抓住"未初始化就累加"
static float run(Fn fn, const std::vector<float>& h) {
  int n = int(h.size());
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.upload(h); out.fill(NAN); ws.fill(NAN);
  fn(in.get(), n, out.get(), ws.get(), nullptr);
  hk::Result tmp;
  if (!out.guards_intact(tmp) || !ws.guards_intact(tmp)) return INFINITY;  // 越界写 → 判错
  if (in.download() != h) return INFINITY;
  return out.download()[0];
}

// 全 1 输入：n < 2^24 时任何求和顺序都精确等于 n，可做精确比较，抓住"漏掉/重复元素"
static void ones(hk::Result& R, Fn fn, const char* name) {
  for (int n : {1, 2, 63, 64, 65, 255, 256, 257, 1000, 4097, (1 << 20) + 1, 3 * (1 << 20) + 5}) {
    float g = run(fn, std::vector<float>(n, 1.0f));
    if (g != float(n)) { hk::fail(R, "%s: 全 1 输入 n=%d，期望 %d，得到 %.9g", name, n, n, g); return; }
  }
}

static void randoms(hk::Result& R, Fn fn, const char* name) {
  for (int n : {1000, 1 << 20, (1 << 24) + 3}) {
    auto h = hk::random_floats(n, n);
    double ref = 0, mag = 0;
    for (float x : h) { ref += x; mag += std::fabs(x); }
    float g = run(fn, h);
    double tol = 1e-5 * mag + 1e-6;
    if (!(std::fabs(g - ref) <= tol)) { hk::fail(R, "%s: 随机输入 n=%d，期望 %.9g，得到 %.9g（容差 %.3g）", name, n, ref, g, tol); return; }
  }
}

TEST_CASE(lds_exact_ones, 3)   { ones(R, reduce_sum_lds, "lds"); }
TEST_CASE(lds_random, 2)       { randoms(R, reduce_sum_lds, "lds"); }
TEST_CASE(shfl_exact_ones, 3)  { ones(R, reduce_sum_shfl, "shfl"); }
TEST_CASE(shfl_random, 2)      { randoms(R, reduce_sum_shfl, "shfl"); }
TEST_CASE(repeatable_output, 1) {
  // 连续调用两次必须得到同样的结果（第二次调用时 out 中已有上次的值）
  int n = 1 << 20;
  auto h = hk::random_floats(n, 99);
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.upload(h); out.fill(NAN); ws.fill(NAN);
  reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr);
  float a = out.download()[0];
  reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr);
  float b = out.download()[0];
  if (!(std::fabs(a - b) <= 1e-5 * n)) hk::fail(R, "两次调用结果不同：%.9g vs %.9g（是否把结果累加到了旧的 out 上？）", a, b);
}

constexpr double kReduceRatio = 0.50;  // Step 6 校准后写死
PERF_CASE(shfl_bandwidth, 3) {
  int n = 1 << 26;
  size_t bytes = size_t(n) * sizeof(float);
  hk::GuardedBuffer<float> in(n), out(1), ws(kWorkspaceFloats);
  in.fill(1.0f);
  double copy = hk::copy_bandwidth_gbs(bytes);
  double ms = hk::time_best_ms([&] { reduce_sum_shfl(in.get(), n, out.get(), ws.get(), nullptr); });
  R.metric = bytes / (ms * 1e-3) / 1e9;
  R.unit = "GB/s";
  float g = out.download()[0];
  if (!(std::fabs(g - double(n)) <= 1e-6 * n)) { hk::fail(R, "结果错误：期望 %d，得到 %.9g", n, g); return; }
  if (R.metric < kReduceRatio * copy)
    hk::fail(R, "读带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kReduceRatio * copy, kReduceRatio * 100, copy);
}
HK_MAIN()
```

注：性能用例 n=2^26 超过 float 能精确表示的整数范围（2^24），所以这里用相对容差 1e-6 比较，而不是精确比较。

- [ ] **Step 2: 参考答案 `solutions/lab1-hip-basics/a3-reduction/reduction.hip`**

```cpp
#include <hip/hip_runtime.h>

constexpr int kWorkspaceFloats = 4096;
constexpr int kBlock = 256;
constexpr int kMaxBlocks = 1024;  // ≤ kWorkspaceFloats

// ---------------- LDS 树形归约 ----------------
__device__ float block_reduce_lds(float v) {
  __shared__ float s[kBlock];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  return s[0];  // 所有线程都能读到
}

// ---------------- wavefront shuffle 归约 ----------------
__device__ float wave_reduce(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
  return v;  // lane 0 持有整个 wavefront 的和
}

__device__ float block_reduce_shfl(float v) {
  __shared__ float partial[kBlock / 64];
  int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  v = wave_reduce(v);
  if (lane == 0) partial[wave] = v;
  __syncthreads();
  v = (threadIdx.x < kBlock / warpSize) ? partial[threadIdx.x] : 0.0f;
  if (wave == 0) v = wave_reduce(v);
  return v;  // thread 0 持有结果
}

template <bool kShfl>
__global__ void partial_sum_kernel(const float* in, int n, float* dst) {
  float acc = 0.0f;
  for (int i = blockIdx.x * kBlock + threadIdx.x; i < n; i += gridDim.x * kBlock) acc += in[i];
  float s = kShfl ? block_reduce_shfl(acc) : block_reduce_lds(acc);
  if (threadIdx.x == 0) dst[blockIdx.x] = s;
}

template <bool kShfl>
static void reduce_two_pass(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  int blocks = (n + kBlock - 1) / kBlock;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  partial_sum_kernel<kShfl><<<blocks, kBlock, 0, stream>>>(in, n, ws);      // 第 1 遍：每 block 一个部分和
  partial_sum_kernel<kShfl><<<1, kBlock, 0, stream>>>(ws, blocks, out);     // 第 2 遍：汇总部分和
}

void reduce_sum_lds(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<false>(in, n, out, ws, stream);
}
void reduce_sum_shfl(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<true>(in, n, out, ws, stream);
}
```

- [ ] **Step 3: 学生骨架（难度 2）**

保留：`kWorkspaceFloats`、`kBlock`、`kMaxBlocks`、`partial_sum_kernel` 模板与 `reduce_two_pass`、两个 launch 函数（两遍结构直接给出，并在注释中画出数据流：`in →(多 block)→ ws[0..blocks) →(1 block)→ out`）。挖空：
- `// TODO(a3-1): partial_sum_kernel 中每个线程用 grid-stride loop 累加自己负责的元素（复习 a1 Part B）`
- `// TODO(a3-2): block_reduce_lds：用 __shared__ 数组做树形归约，每一层之后需要什么？（讲义 §4.2）` —— 函数体只剩 `return 0.0f;`
- `// TODO(a3-3): wave_reduce：用 __shfl_down 在一个 wavefront 内归约。注意 AMD 的 wavefront 是 64 线程，用 warpSize 而不是写死 32（讲义 §4.3）`
- `// TODO(a3-4): block_reduce_shfl：先 wave 内归约，再把每个 wave 的结果放进 LDS，最后由第 0 个 wave 再归约一次（讲义 §4.4）`

- [ ] **Step 4: mutants**

- `m1_drop_tail.hip`：grid-stride 循环条件写成 `i < (n / kBlock) * kBlock` → 全 1 用例 n=65 FAIL。
- `m2_atomic_no_init.hip`：第 2 遍改为在第 1 遍中直接 `atomicAdd(out, s)`（不清零 out）→ out 预置 NaN，FAIL。
- `m3_single_block.hip`：`blocks = 1` → 结果正确，`shfl_bandwidth` FAIL（依赖校准）。
- `m4_shfl_width32.hip`：`wave_reduce` 写死 `for (int offset = 16; ...)` → 每个 wave 丢掉后 32 个 lane 的部分和，全 1 用例 FAIL。

- [ ] **Step 5: 校准 `kReduceRatio`**

同 Task 6 Step 6 的方法：对 solution 与 `m3_single_block.hip` 各测 5 次，取 `kReduceRatio = sqrt(S×N)`，要求 S/N ≥ 2.25；不满足则按 Task 6 的兜底处理并记录。

- [ ] **Step 6: README**

Hint 1：为什么需要两遍（block 之间没有 `__syncthreads`）。Hint 2：树形归约每层活跃线程减半；shuffle 让同一 wavefront 的线程不经过 LDS 直接交换寄存器。书面问题：
1. `m4` 式的错误（offset 从 16 开始）在 NVIDIA GPU 上是对的，在 MI308X 上为什么错？
2. 为什么随机输入只能用容差比较，而全 1 输入可以精确比较？对 2^25 个 1 求和时，单线程顺序累加的结果是多少？为什么？
3. 第 2 遍只用 1 个 block，会成为瓶颈吗？用 `rocprofv3 --kernel-trace` 看两个 kernel 各自耗时。
4. （思考）能否只用一个 kernel 完成归约？（提示：atomic，或"最后一个完成的 block 负责汇总"——这是后面 mega kernel 中常见的跨 block 同步手法的雏形）

- [ ] **Step 7: 验证与 Commit**

Run: `cd lab1-hip-basics && make verify` → Expected: `VERIFY OK`。
```bash
git add lab1-hip-basics/a3-reduction solutions/lab1-hip-basics/a3-reduction
git commit -m "feat(lab1): a3 two-pass reduction with LDS and wave64 shuffle"
```

---
### Task 8: Lab 1 · a4 GEMM（naive → LDS 分块，Lab 1 收尾）

**学习目标：** 把前三个 Assignment 的技能组合起来：2D 网格、LDS 分块复用、同步、边界处理；第一次接触"计算强度 / 访存复用"的概念，为 Lab 4 的 MFMA GEMM 打基础。tiled 部分是**难度 3（自主设计）**：只给接口。

**Files:**
- Create: `lab1-hip-basics/a4-gemm/{gemm.hip,test.cpp,README.md}`
- Create: `solutions/lab1-hip-basics/a4-gemm/gemm.hip` + `mutants/{m1_naive_b_colmajor.hip,m2_tiled_drop_k_tail.hip,m3_tiled_is_naive.hip,m4_tiled_no_bounds_load.hip}`

**Interfaces:**
```cpp
// C[M×N] = A[M×K] · B[K×N]，全部行主序 fp32
void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream);
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab1-hip-basics/a4-gemm/test.cpp`**

```cpp
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

constexpr double kTiledGflops = 1000.0;  // Step 5 校准后写死
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
```

注：GEMM 没有像拷贝带宽那样简单的同进程基准，因此用绝对阈值；grade.py 对性能用例最多重试 3 次并选最空闲 GPU。

- [ ] **Step 2: 参考答案 `solutions/lab1-hip-basics/a4-gemm/gemm.hip`**

```cpp
#include <hip/hip_runtime.h>

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列：相邻线程访问 B、C 的相邻元素 → 合并访存
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[size_t(i) * K + k] * B[size_t(k) * N + j];
  C[size_t(i) * N + j] = acc;
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((N + 63) / 64, (M + 3) / 4);
  gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// 分块：block 计算 C 的 TILE×TILE 子块；256 线程 = 32 列 × 8 行，每线程计算同一列上 TILE/8 = 4 个元素。
constexpr int TILE = 32;
constexpr int TY = 8;
constexpr int RPT = TILE / TY;  // rows per thread

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row0 = blockIdx.y * TILE, col = blockIdx.x * TILE + tx;
  float acc[RPT] = {};
  for (int k0 = 0; k0 < K; k0 += TILE) {
    for (int r = 0; r < RPT; ++r) {
      int lr = ty + r * TY;
      int ar = row0 + lr, ac = k0 + tx;
      As[lr][tx] = (ar < M && ac < K) ? A[size_t(ar) * K + ac] : 0.0f;  // 越界补 0
      int br = k0 + lr;
      Bs[lr][tx] = (br < K && col < N) ? B[size_t(br) * N + col] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < TILE; ++kk) {
      float b = Bs[kk][tx];
      for (int r = 0; r < RPT; ++r) acc[r] += As[ty + r * TY][kk] * b;
    }
    __syncthreads();  // 下一轮覆盖 As/Bs 前，所有线程必须用完当前 tile
  }
  for (int r = 0; r < RPT; ++r) {
    int i = row0 + ty + r * TY;
    if (i < M && col < N) C[size_t(i) * N + col] = acc[r];
  }
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  dim3 block(TILE, TY);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
```

- [ ] **Step 3: 学生骨架**

- naive（难度 1）：launch 函数完整给出；kernel 体留 `// TODO(a4-1): 每个线程计算 C 的一个元素（讲义 §5.1）。想想：让相邻线程对应相邻的 j 还是相邻的 i？`
- tiled（难度 3）：只有
```cpp
// TODO(a4-2): 自主设计 LDS 分块 GEMM（讲义 §5.2–5.4）
//   你需要决定：tile 尺寸、block 形状、每个线程算几个输出、LDS 怎么摆放、K 方向怎么循环、边界怎么处理。
//   可复用 a2 学到的 padding、a3 学到的同步。目标：tiled_gflops 用例的性能门槛（见 test.cpp 中 kTiledGflops）。
//   设计完成后，在 README 的"设计说明"处用 3–5 句话写下你的选择和理由。
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  // >>> 你的代码 <<<
}
```

- [ ] **Step 4: mutants**

- `m1_naive_b_colmajor.hip`：naive 用 `B[size_t(j) * K + k]` → 非方阵 FAIL。
- `m2_tiled_drop_k_tail.hip`：`for (int k0 = 0; k0 + TILE <= K; k0 += TILE)` → K=65/257/1 FAIL。
- `m3_tiled_is_naive.hip`：`launch_gemm_tiled` 调用 naive kernel → `tiled_gflops` FAIL（依赖校准）。
- `m4_tiled_no_bounds_load.hip`：tile 加载去掉 `(ar < M && ac < K)` 与 `(br < K && col < N)` 判断（直接读）→ 读到下一行或哨兵区的非零值，K=65 等 FAIL。

- [ ] **Step 5: 校准 `kTiledGflops`**

同 Task 6 Step 6：solution 与 m3 各测 5 次，记录 GFLOPS 的最小值 S_min 与 naive 最大值 N_max；要求 S_min ≥ 2.25 × N_max；`kTiledGflops = sqrt(S_min × N_max)` 取整到 100。不满足时先改进参考答案（例如每线程 4×4 输出的寄存器分块），仍不满足按兜底处理并记录。

- [ ] **Step 6: README**

Hint 1：naive 中每个 A、B 元素被从全局内存读了多少次？Hint 2：把 A 的一行块和 B 的一列块搬到 LDS，block 内所有线程复用；K 方向分段循环，每段前后各需一次同步。书面问题：
1. 计算 naive 与 tiled（TILE=32）两种实现每计算一次 FMA 需要的全局内存读取字节数（计算强度）。
2. 用你的 GFLOPS 与 MI308X fp32 向量峰值（讲义 §1 表格）比较，差距主要来自哪里？（铺垫 Lab 4：MFMA 矩阵核心）
3. 设计说明（3–5 句话）。

- [ ] **Step 7: 验证与 Commit**

Run: `cd lab1-hip-basics && make verify` → Expected: `VERIFY OK`（a1–a4 全部）。
```bash
git add lab1-hip-basics/a4-gemm solutions/lab1-hip-basics/a4-gemm
git commit -m "feat(lab1): a4 naive and LDS-tiled GEMM"
```

---
### Task 9: Lab 1 讲义 `lab1-hip-basics/tutorial.html` + `answers.md`

**Files:**
- Create: `lab1-hip-basics/tutorial.html`（从 `common/tutorial-template.html` 复制）, `lab1-hip-basics/answers.md`

章节与小节锚点（各 Assignment 的 TODO 注释引用这些编号，**编号必须一致**）：

1. **硬件心智模型**：MI308X 层级（XCD → CU → SIMD → wavefront）；内存层级（寄存器 / LDS / L2 / Infinity Cache / HBM）；`table.facts`：本机实测 CU 数、wave64、LDS 64 KiB、HBM 容量；HBM 峰值带宽与 fp32 向量峰值按 CDNA3 白皮书公式现算（实现时用 `rocminfo` 数据代入并注明出处）。
2. **从 vector add 开始**：§2.1 SIMT 与 wavefront；§2.2 kernel 与 launch；§2.3 全局下标与边界检查；§2.4 选 block 大小（64 的倍数、occupancy 初步）；§2.5 grid-stride loop（widget：调 n、grid、block，动画显示每个线程依次处理哪些元素）。→ assignment-card a1
3. **访存模式与转置**：§3.1 合并访存（SVG：一个 wavefront 64 个地址落在哪些 cache line）；§3.2 2D 网格与 naive 转置；§3.3 为什么 naive 转置的写是非合并的；§3.4 LDS 是什么、`__shared__` 与生命周期；§3.5 `__syncthreads` 与"同步屏障"；§3.6 bank conflict 与 padding（widget：选择 `TILE+pad`，高亮按列读取时 64 个 lane 命中的 bank，显示冲突度）；§3.7 用带宽衡量 kernel、读懂 `lds_bandwidth` 用例。→ assignment-card a2
4. **归约**：§4.1 两遍归约结构（SVG 数据流）；§4.2 LDS 树形归约（逐层动画）；§4.3 `__shfl_down` 与 wave64（强调 CUDA 资料中的 32 在这里是 64）；§4.4 block 级 shuffle 归约；§4.5 浮点求和误差与为什么测试用全 1 输入做精确比较。→ assignment-card a3
5. **GEMM**：§5.1 naive GEMM 与线程到输出的映射；§5.2 计算强度与数据复用（图：A 行块 × B 列块）；§5.3 LDS 分块与 K 方向循环、两次同步；§5.4 每线程多个输出、边界补零；§5.5 展望：MFMA（Lab 4）。→ assignment-card a4
6. **调试与剖析小抄**：`make run A=.. CASE=..`、`printf` in kernel、`rocprofv3 --kernel-trace --stats`、`HIP_LAUNCH_BLOCKING=1`。

- [ ] **Step 1: 编写讲义**：所有示例代码片段必须与 Assignment 的接口一致，但**不得直接给出 TODO 处的完整答案**（可给出原理与等价的伪代码 / 其他问题上的同类代码，例如讲 grid-stride 时用 `memset` 示例而非 saxpy）。
- [ ] **Step 2: 编写 3 个 widget**（grid-stride 映射、bank conflict 可视化、树形归约动画），通过 `TUT.registerWidget` 注册，写在页面底部 `<script>` 中。
- [ ] **Step 3: `answers.md`**：汇总 a1–a4 README 中的书面问题，每题后留作答区。
- [ ] **Step 4: 浏览器验证**（同 Task 3 Step 4：无 console error、浅/深色截图、390px 宽无横向滚动、3 个 widget 交互后 DOM 有变化）。
- [ ] **Step 5: 交叉核对**：`grep -o "讲义 §[0-9.–]*" lab1-hip-basics/a*/*.hip lab1-hip-basics/a*/README.md`，逐个确认 tutorial.html 中存在对应小节。
- [ ] **Step 6: Commit**

```bash
git add lab1-hip-basics/tutorial.html lab1-hip-basics/answers.md
git commit -m "docs(lab1): HIP basics tutorial"
```

---

### Task 10: 审查流程 + Lab 0/Lab 1 独立审查

**Files:**
- Create: `docs/reviews/PROMPTS.md`, `docs/reviews/lab0.md`, `docs/reviews/lab1.md`

- [ ] **Step 1: 写 `docs/reviews/PROMPTS.md`**（三个审查 agent 的提示词，`{LAB}` 为参数，调用时替换为具体 Lab 目录名）

```markdown
# 审查提示词（每个 Lab 完成后并行派发 3 个 subagent）

## R1 学生视角走查
你是一名会用 C++、了解 GPU 大致概念、但没写过 HIP kernel 的学生（CK-Tile 模板元编程不熟）。仓库根目录 /mnt/raid0/WangTianle/workspace/HIP-tutorial。
**禁止读取 solutions/ 目录与任何 mutants。** 只读 {LAB}/tutorial.html（可用浏览器或直接读 HTML）、{LAB}/aK-*/README.md 与骨架文件。
任务：按顺序在一个临时拷贝中完成每个 Assignment（`cp -r {LAB} /tmp/r1-{LAB}` 后在拷贝中做；`common/` 用原路径），用 `make grade` 检查。
报告（中文）：每个 Assignment 卡住的地方与原因（讲义缺了什么、Hint 不够、知识跳跃、TODO 说明有歧义）；讲义中引用但不存在的小节；判分输出是否帮你定位问题；最终得分。不要修改原仓库。

## R2 判分对抗审查
你的目标是找出判分漏洞：写出"看起来对但实际错误"的实现，看 `{LAB}` 的测试能否抓住。可以读 solutions/ 和 test.cpp。
对每个 Assignment 至少尝试 4 种不同错误（越界、漏边界、错误下标、数据竞争、只对特定尺寸成立、偷懒的性能作弊如不计算直接返回、写错误位置后又被覆盖等）。
用 `python3 common/grade.py {LAB} --only aK` 在临时目录的拷贝上测试（不要改原仓库文件）。
同时检查是否存在"正确实现被判错"的情况（例如合理的不同 block 尺寸、不同求和顺序、不同 tile 选择）——至少写 2 个与参考答案结构不同的正确实现并确认满分。
报告：漏网的错误实现（附代码）、被误判的正确实现（附代码）、建议新增的测试/mutant。

## R3 技术准确性审查
核对 {LAB}/tutorial.html、README 与代码注释中的技术事实：gfx942/CDNA3 硬件参数（CU 数、wave 大小、LDS 大小与 bank 数、缓存层级、峰值带宽/算力的计算）、HIP API 语义、CK-Tile API 用法（对照 /opt/rocm/include/ck_tile 源码）。可用 rocminfo、hipcc 写小程序实测。
报告：每条错误/不严谨陈述的位置、原文、更正及依据（源码路径或实测输出）。
```

- [ ] **Step 2: 对 lab0-env 与 lab1-hip-basics 各派发 R1/R2/R3**（共 6 个 subagent，并行，同一消息中发出）。

- [ ] **Step 3: 处理审查结果**：对每条发现：确认属实 → 修复（新增测试/mutant、改讲义/README）→ 重新 `make verify`；不属实 → 在报告中写明理由。R2 发现的每个漏网错误实现都要加为新 mutant。

- [ ] **Step 4: 写 `docs/reviews/lab0.md`、`docs/reviews/lab1.md`**：发现列表、处理结果、校准数据（各性能用例的 S、N、阈值）、最终 `make verify` 输出摘要。

- [ ] **Step 5: 复查**：仅对有修复的部分重新派发对应审查（R1 或 R2）一次，确认问题关闭。

- [ ] **Step 6: Commit**

```bash
git add -A lab0-env lab1-hip-basics solutions docs/reviews
git commit -m "review(lab0, lab1): independent review fixes and reports"
```

---
### Task 11: Lab 2 · a1 CK-Tile 所需 C++ 速成

**学习目标：** 读懂 CK-Tile 代码所需的 C++：`constexpr` 函数、变参模板与折叠表达式、`number<N>`（把值编码进类型）、`sequence<...>`、`tuple` 与 `.at(number<I>{})`、`static_for`（编译期展开循环，循环变量是 `number<I>` 类型）、`if constexpr`、泛型 lambda。全部用 CK-Tile 自己的类型练习（已在本机实测可编译）。

**Files:**
- Create: `lab2-cktile-intro/Makefile`（`include ../common/lab.mk`）
- Create: `lab2-cktile-intro/a1-cpp-meta/{cpp_meta.hip,test.cpp,README.md}`
- Create: `solutions/lab2-cktile-intro/a1-cpp-meta/cpp_meta.hip` + `mutants/{m1_product_zero_init.hip,m2_strides_colmajor.hip,m3_vec_width_bits.hip}`

**Interfaces:**
```cpp
#include "ck_tile/core.hpp"
using ck_tile::index_t;
// E1：sequence 中所有元素之积；空 sequence 返回 1
template <index_t... Is> constexpr index_t product(ck_tile::sequence<Is...>);
// E2：紧密排列的行主序 stride，如 <4,8,2> → {16,2,1}
template <index_t... Ls> constexpr auto row_major_strides(ck_tile::sequence<Ls...>);  // 返回 ck_tile::array<index_t, sizeof...(Ls)>
// E3：设备端用 static_for 展开写 out[i] = i*i（i < N）
template <index_t N> __device__ void fill_squares(float* out);
// E4：一次 16 字节向量访存能装多少个 T；double 特殊返回 2（演示 if constexpr 分支）
template <class T> constexpr index_t vec16();
// E5：tuple 中所有元素（可异构数值类型）转为 double 后求和
template <class... Ts> constexpr double tuple_sum(const ck_tile::tuple<Ts...>& t);
// E6：泛型 lambda：两个 number<> 相加，结果仍是 number<>（编译期）
// constexpr auto add_numbers = [](auto a, auto b) { ... };
```

- [ ] **Step 1: 测试 `lab2-cktile-intro/a1-cpp-meta/test.cpp`**

```cpp
#include "harness.hpp"
#include STUDENT_SRC
#include <type_traits>
using namespace ck_tile;

__global__ void squares_kernel8(float* o) { fill_squares<8>(o); }
__global__ void squares_kernel1(float* o) { fill_squares<1>(o); }

TEST_CASE(e1_product, 2) {
  constexpr index_t a = product(sequence<2, 3, 4>{});   // 必须能在编译期求值
  if (a != 24) hk::fail(R, "product<2,3,4> 期望 24，得到 %d", a);
  if (product(sequence<>{}) != 1) hk::fail(R, "空 sequence 的积应为 1，得到 %d", product(sequence<>{}));
  if (product(sequence<7>{}) != 7) hk::fail(R, "product<7> 应为 7");
}

TEST_CASE(e2_row_major_strides, 3) {
  constexpr auto s = row_major_strides(sequence<4, 8, 2>{});
  static_assert(std::is_same_v<std::remove_cv_t<decltype(s)>, array<index_t, 3>>, "返回类型应为 ck_tile::array<index_t, 3>");
  if (s[number<0>{}] != 16 || s[number<1>{}] != 2 || s[number<2>{}] != 1)
    hk::fail(R, "<4,8,2> 期望 {16,2,1}，得到 {%d,%d,%d}", s[number<0>{}], s[number<1>{}], s[number<2>{}]);
  auto t = row_major_strides(sequence<5>{});
  if (t[number<0>{}] != 1) hk::fail(R, "<5> 期望 {1}");
  auto u = row_major_strides(sequence<3, 1, 7, 2>{});
  if (u[number<0>{}] != 14 || u[number<1>{}] != 14 || u[number<2>{}] != 2 || u[number<3>{}] != 1)
    hk::fail(R, "<3,1,7,2> 期望 {14,14,2,1}");
}

TEST_CASE(e3_static_for_device, 2) {
  hk::GuardedBuffer<float> o(8), p(1);
  o.fill(-1); p.fill(-1);
  squares_kernel8<<<1, 1>>>(o.get());
  squares_kernel1<<<1, 1>>>(p.get());
  std::vector<double> ref = {0, 1, 4, 9, 16, 25, 36, 49};
  if (hk::check_close(R, o.download(), ref, 0, 0, "fill_squares<8>") && o.guards_intact(R))
    if (hk::check_close(R, p.download(), {0.0}, 0, 0, "fill_squares<1>")) p.guards_intact(R);
}

TEST_CASE(e4_if_constexpr, 2) {
  constexpr index_t f = vec16<float>(), h = vec16<half_t>(), i8 = vec16<int8_t>(), d = vec16<double>();
  if (f != 4 || h != 8 || i8 != 16 || d != 2)
    hk::fail(R, "期望 float=4 half=8 int8=16 double=2，得到 %d %d %d %d", f, h, i8, d);
}

TEST_CASE(e5_tuple_sum, 2) {
  constexpr double s = tuple_sum(make_tuple(1, 2.5f, 3.0));
  if (s != 6.5) hk::fail(R, "(1, 2.5f, 3.0) 之和期望 6.5，得到 %g", s);
  if (tuple_sum(make_tuple(int8_t(-3))) != -3.0) hk::fail(R, "单元素 tuple 求和错误");
}

TEST_CASE(e6_compile_time_lambda, 2) {
  using T = decltype(add_numbers(number<3>{}, number<4>{}));
  if (!std::is_same_v<T, number<7>>) hk::fail(R, "add_numbers(number<3>, number<4>) 的类型应为 number<7>，结果需在类型中");
  using U = decltype(add_numbers(number<0>{}, number<-5>{}));
  if (!std::is_same_v<U, number<-5>>) hk::fail(R, "add_numbers(number<0>, number<-5>) 应为 number<-5>");
}
HK_MAIN()
```

注意：`constexpr index_t a = product(...)` 等要求学生实现必须是 constexpr——若学生写成非 constexpr，**整个 test 编译失败**，判 0 分并显示编译错误。这是有意设计（CK-Tile 中非 constexpr 同样无法编译），README 中明确提醒"编译失败时先看是否丢了 constexpr"。骨架中所有函数都已是 constexpr 且可编译。

- [ ] **Step 2: 参考答案 `solutions/lab2-cktile-intro/a1-cpp-meta/cpp_meta.hip`**

```cpp
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include <type_traits>
using ck_tile::index_t;

template <index_t... Is>
constexpr index_t product(ck_tile::sequence<Is...>) {
  return (index_t{1} * ... * Is);  // 折叠表达式；空包时结果为初值 1
}

template <index_t... Ls>
constexpr auto row_major_strides(ck_tile::sequence<Ls...>) {
  constexpr index_t n = sizeof...(Ls);
  constexpr index_t lens[n > 0 ? n : 1] = {Ls...};
  ck_tile::array<index_t, n> s{};
  index_t acc = 1;
  for (index_t i = n - 1; i >= 0; --i) {  // 从最内层维度往外累乘
    s(i) = acc;
    acc *= lens[i];
  }
  return s;
}

template <index_t N>
__device__ void fill_squares(float* out) {
  ck_tile::static_for<0, N, 1>{}([&](auto i) {  // i 的类型是 number<0>, number<1>, ...
    constexpr index_t v = i.value;                // 编译期常量
    out[v] = float(v * v);
  });
}

template <class T>
constexpr index_t vec16() {
  if constexpr (std::is_same_v<T, double>) return 2;
  else return 16 / sizeof(T);
}

template <class... Ts>
constexpr double tuple_sum(const ck_tile::tuple<Ts...>& t) {
  double s = 0;
  ck_tile::static_for<0, sizeof...(Ts), 1>{}([&](auto i) { s += double(t.at(i)); });
  return s;
}

constexpr auto add_numbers = [](auto a, auto b) {
  return ck_tile::number<decltype(a)::value + decltype(b)::value>{};
};
```

- [ ] **Step 3: 学生骨架**：每个函数保留签名与 `constexpr`，函数体为可编译的错误占位：`return 0;`、`return ck_tile::array<index_t, sizeof...(Ls)>{};`、`fill_squares` 空、`return 1;`、`return 0.0;`、`return ck_tile::number<0>{};`。每个函数上方写 `// TODO(a1-En)` 与讲义小节（§2.1–2.6），并附一个**同类但不同**的小例子（例如 E1 给出"求和"的折叠表达式例子，学生改成乘积；E2 给出列主序的口头定义，不给代码）。

- [ ] **Step 4: mutants**

- `m1_product_zero_init.hip`：折叠初值写成 `index_t{0}` → `e1_product` FAIL。
- `m2_strides_colmajor.hip`：从最外层往内累乘（列主序）→ `e2_row_major_strides` FAIL。
- `m3_vec_width_bits.hip`：`vec16` 返回 `16 / (sizeof(T) * 8)`（把字节和比特弄混）→ `e4_if_constexpr` FAIL。

- [ ] **Step 5: README**：书面问题：
1. `static_for` 的循环变量为什么不能是普通 `int`？（提示：`number<I>` 能用于模板参数）
2. `number<3>{} + number<4>{}` 在 CK-Tile 中结果的类型是什么？去 `/opt/rocm/include/ck_tile/core/numeric/integral_constant.hpp` 找到答案并引用相关代码行。
3. 为什么 CK-Tile 要把 tile 尺寸等参数放进类型里，而不是作为普通函数参数？

- [ ] **Step 6: 验证与 Commit**

Run: `cd lab2-cktile-intro && make verify` → Expected: `VERIFY OK`。
```bash
git add lab2-cktile-intro/Makefile lab2-cktile-intro/a1-cpp-meta solutions/lab2-cktile-intro/a1-cpp-meta
git commit -m "feat(lab2): a1 C++ metaprogramming warm-up with ck_tile types"
```

---
### Task 12: Lab 2 · a2 用 CK-Tile 写 vector add

**学习目标：** CK-Tile 的"四件套"：tensor view（全局内存的有形状视图）→ tile distribution（谁负责 tile 中的哪些元素）→ tile window（视图上的一个窗口）→ `load_tile` / `sweep_tile` / `store_tile`。1D 情形下理解 `tile_distribution_encoding` 的 H/P/Y 维与"2 个 P 维 = (warp id, lane id)"。与 Lab 1 a1 做同一件事，便于对照。

**已实测（/tmp 探测，ROCm 7.2）：** 下方参考答案可编译；n ∈ {1, 3, 1023, 1024, 4097, 2^20} 结果正确且无越界写；若 encoding 只有 1 个 P 维，则每个 block 只有前 64 个线程生效（n=1024 时 768 个元素错误）。

**Files:**
- Create: `lab2-cktile-intro/a2-ck-vector-add/{ck_vector_add.hip,test.cpp,README.md}`
- Create: `solutions/lab2-cktile-intro/a2-ck-vector-add/ck_vector_add.hip` + `mutants/{m1_single_p_dim.hip,m2_origin_missing_tile.hip,m3_single_block_loop.hip}`

**Interfaces:**
```cpp
void launch_ck_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab2-cktile-intro/a2-ck-vector-add/test.cpp`**

```cpp
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

constexpr double kAddRatio = 0.60;  // Step 5 校准后写死
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
  if (R.metric < kAddRatio * copy)
    hk::fail(R, "带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kAddRatio * copy, kAddRatio * 100, copy);
}
HK_MAIN()
```

- [ ] **Step 2: 参考答案 `solutions/lab2-cktile-intro/a2-ck-vector-add/ck_vector_add.hip`**

```cpp
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

constexpr index_t kWarps = 4;               // 每个 block 4 个 wavefront
constexpr index_t kLanes = 64;              // 每个 wavefront 64 个 lane
constexpr index_t kVec = 4;                 // 每个线程连续处理 4 个 float（16 字节向量访存）
constexpr index_t kBlock = kWarps * kLanes; // 256 线程
constexpr index_t kTile = kBlock * kVec;    // 每个 block 处理 1024 个元素

__global__ void ck_vector_add_kernel(const float* a, const float* b, float* c, index_t n) {
  // 1) tensor view：给裸指针加上形状 (n)、stride (1)；越界访问由 buffer 指令按长度 n 屏蔽
  auto va = make_naive_tensor_view<address_space_enum::global>(a, make_tuple(n), make_tuple(1), number<kVec>{}, number<1>{});
  auto vb = make_naive_tensor_view<address_space_enum::global>(b, make_tuple(n), make_tuple(1), number<kVec>{}, number<1>{});
  auto vc = make_naive_tensor_view<address_space_enum::global>(c, make_tuple(n), make_tuple(1), number<kVec>{}, number<1>{});

  // 2) tile distribution：把长度 kTile 的 1D tile 拆成 H = (kWarps, kLanes, kVec)
  //    P0 → H 的第 0 段（warp id），P1 → 第 1 段（lane id），Y → 第 2 段（每线程的 kVec 个元素）
  constexpr auto enc = tile_distribution_encoding<
      sequence<>,                                   // R：无复制维
      tuple<sequence<kWarps, kLanes, kVec>>,        // H：一个 X 维被拆成 3 段
      tuple<sequence<1>, sequence<1>>,              // P0、P1 都映射到 X 维 1（即 H 的第 1 个元组元素）
      tuple<sequence<0>, sequence<1>>,              // ……的第 0 段、第 1 段
      sequence<1>,                                  // Y 映射到 X 维 1
      sequence<2>>{};                               // ……的第 2 段
  constexpr auto dist = make_static_tile_distribution(enc);

  // 3) tile window：本 block 负责 [blockIdx.x * kTile, +kTile)
  const index_t origin = static_cast<index_t>(blockIdx.x) * kTile;
  auto wa = make_tile_window(va, make_tuple(number<kTile>{}), {origin}, dist);
  auto wb = make_tile_window(vb, make_tuple(number<kTile>{}), {origin}, dist);
  auto wc = make_tile_window(vc, make_tuple(number<kTile>{}), {origin}, dist);

  // 4) load → 逐元素计算 → store
  auto ta = load_tile(wa);
  auto tb = load_tile(wb);
  auto tc = make_static_distributed_tensor<float>(dist);
  sweep_tile(tc, [&](auto idx) { tc(idx) = ta[idx] + tb[idx]; });
  store_tile(wc, tc);
}

void launch_ck_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  const int grid = (n + kTile - 1) / kTile;
  ck_vector_add_kernel<<<grid, kBlock, 0, stream>>>(a, b, c, n);
}
```

- [ ] **Step 3: 学生骨架（难度 2）**

保留常量定义、`va` 的完整构造（作为示例，并 `(void)va;` 避免未使用警告），挖空：
- `// TODO(a2-1): 仿照 va 构造 vb、vc（讲义 §3.1）`
- `// TODO(a2-2): 写出 tile_distribution_encoding 的 6 个模板参数并 make_static_tile_distribution（讲义 §3.2）。先在纸上画出 kTile 个元素如何分给 kWarps×kLanes 个线程。`（骨架中 kernel 只有 `va` 的构造，其余 TODO 处均为空——不给"能编译的错误占位 encoding"，因为某些错误 encoding 恰好能算出正确结果，会误导学生；空 kernel 可编译，原样提交时输出保持 NaN 而全部 FAIL）
- `// TODO(a2-3): 计算本 block 的窗口原点并创建 3 个 tile window（讲义 §3.3）。注意 blockIdx.x 是 unsigned，需要 static_cast<index_t>`
- `// TODO(a2-4): load、sweep_tile 中逐元素相加、store（讲义 §3.4）`
- `launch` 中 grid 计算给出。

- [ ] **Step 4: mutants**

- `m1_single_p_dim.hip`：encoding 改为 `tuple<sequence<kBlock, kVec>>`、`tuple<sequence<1>>`、`tuple<sequence<0>>`、Y→`sequence<1>`/`sequence<1>`（单 P 维，实测只有 64 线程生效）→ `one_tile` FAIL。
- `m2_origin_missing_tile.hip`：`origin = blockIdx.x * kBlock` → `many_tiles` FAIL。
- `m3_single_block_loop.hip`：grid=1，kernel 内 `for (tile = 0; tile * kTile < n; ++tile)` 移动窗口（`move_tile_window`）→ 正确但慢，`bandwidth` FAIL（依赖校准）。

- [ ] **Step 5: 校准 `kAddRatio`**：同 Task 6 Step 6（solution 对 m3）。

- [ ] **Step 6: README**：Hint 1：先写出 H=(4,64,4) 时线程 (warp=1, lane=2) 负责的元素下标；Hint 2：P 维个数决定 CK-Tile 用 `get_lane_id()` 还是 `(get_warp_id(), get_lane_id())`（指出源码 `core/tensor/tile_distribution.hpp` 中的对应行，让学生自己读）。书面问题：
1. 把 kVec 改成 1 或 8，`bandwidth` 用例结果怎样变化？为什么？
2. n=1023 时最后一个 tile 不满，为什么不需要写 `if (i < n)`？越界的 load 返回什么？（读 `buffer_view.hpp` 找答案）
3. 比较本作业与 Lab 1 a1 的代码：CK-Tile 把哪些"手写下标计算"变成了类型层面的描述？

- [ ] **Step 7: 验证与 Commit**

Run: `cd lab2-cktile-intro && make verify` → Expected: `VERIFY OK`。
```bash
git add lab2-cktile-intro/a2-ck-vector-add solutions/lab2-cktile-intro/a2-ck-vector-add
git commit -m "feat(lab2): a2 CK-Tile vector add"
```

---

### Task 13: Lab 2 讲义 + 审查

**Files:**
- Create: `lab2-cktile-intro/tutorial.html`, `lab2-cktile-intro/answers.md`, `docs/reviews/lab2.md`

章节：
1. **为什么需要 CK-Tile**：手写 kernel 的痛点（下标、向量化、LDS 布局、MFMA 布局）；CK-Tile 的分层（core：tensor/distribution/window；ops：gemm/fmha/...）；源码地图（`/opt/rocm/include/ck_tile` 目录树）。
2. **C++ 速成**：§2.1 constexpr 与折叠表达式；§2.2 `array` 与编译期计算；§2.3 `number<N>` 与 `static_for`；§2.4 `if constexpr`；§2.5 `tuple`；§2.6 泛型 lambda 与 `decltype(a)::value`。每节一个"读 CK-Tile 源码"的真实片段（标出文件路径）。→ assignment-card a1
3. **第一个 CK-Tile kernel**：§3.1 tensor view 与 buffer 寻址（越界屏蔽）；§3.2 tile distribution encoding（R/H/P/Y 的含义；1D 情形；widget：输入 Warps/Lanes/Vec，显示 1D tile 中每个元素属于哪个 (warp, lane, y)，并可切换"1 个 P 维"看只有 64 个线程生效的现象）；§3.3 tile window 与原点；§3.4 load_tile / distributed tensor / sweep_tile / store_tile；§3.5 与 Lab 1 a1 的逐行对照表。→ assignment-card a2
4. **读源码练习指引**：按顺序读 `tensor_view.hpp` → `tile_window.hpp` → `load_tile.hpp`，各给 3 个"带着问题读"的问题。

- [ ] **Step 1: 编写讲义与 widget**（要求同 Task 9 Step 1–2；所有 CK-Tile 代码片段必须在本机能编译——实现时把每个片段抽出编译一次）
- [ ] **Step 2: `answers.md`**
- [ ] **Step 3: 浏览器验证**（同 Task 3 Step 4）与小节编号交叉核对（同 Task 9 Step 5）
- [ ] **Step 4: 用 `docs/reviews/PROMPTS.md` 派发 R1/R2/R3（`{LAB}=lab2-cktile-intro`），处理结果，写 `docs/reviews/lab2.md`，必要时复查**（流程同 Task 10 Step 3–5）
- [ ] **Step 5: Commit**

```bash
git add lab2-cktile-intro solutions/lab2-cktile-intro docs/reviews/lab2.md
git commit -m "docs(lab2): C++ crash course and CK-Tile intro tutorial, reviewed"
```

---
### Task 14: Lab 3 · a1 layout：tensor descriptor 与坐标变换

**学习目标：** CK-Tile 的 layout 核心：`tensor descriptor` = 一串坐标变换（pass_through、merge、unmerge、xor）把"上层逻辑坐标"映射到"线性偏移"。学会用同一块内存构造不同视图（转置、展平、分块、swizzle），这是之后 LDS 布局与 GEMM tile 划分的基础。以下所有变换已在本机实测（ROCm 7.2）。

**Files:**
- Create: `lab3-cktile-concepts/Makefile`（`include ../common/lab.mk`）
- Create: `lab3-cktile-concepts/a1-layout/{layout.hip,test.cpp,README.md}`
- Create: `solutions/lab3-cktile-concepts/a1-layout/layout.hip` + `mutants/{m1_colmajor_swapped.hip,m2_transpose_wrong_dims.hip,m3_tiled_wrong_order.hip,m4_xor_dims.hip}`

**Interfaces（全部 `CK_TILE_HOST_DEVICE constexpr`，在设备端被调用）:**
```cpp
template <index_t M, index_t N> constexpr auto make_row_major();       // (i,j) → i*N + j
template <index_t M, index_t N> constexpr auto make_col_major();       // (i,j) → j*M + i
template <index_t M, index_t N> constexpr auto make_transposed_view(); // 基于 row-major M×N，上层坐标 (j,i)，j<N, i<M → i*N + j
template <index_t M, index_t N> constexpr auto make_flattened();       // 基于 row-major M×N，上层 1D 坐标 k → k
template <index_t M, index_t N, index_t TM, index_t TN>
constexpr auto make_tiled();                                           // 上层 (I,J,ii,jj) → (I*TM+ii)*N + J*TN + jj
template <index_t M, index_t N> constexpr auto make_xor_swizzled();    // 上层 (i,j) → i*N + (j ^ (i % N))
```

- [ ] **Step 1: 测试 `lab3-cktile-concepts/a1-layout/test.cpp`**

```cpp
#include "harness.hpp"
#include STUDENT_SRC
#include <functional>
using namespace ck_tile;

// 设备端把 descriptor 在所有上层坐标上的偏移写出来，再与 CPU 公式逐一比较
template <class MakeDesc, index_t D0, index_t D1>
__global__ void dump2d(int* out) {
  constexpr auto d = MakeDesc{}();
  for (int i = 0; i < D0; ++i)
    for (int j = 0; j < D1; ++j) out[i * D1 + j] = d.calculate_offset(make_tuple(i, j));
}

template <class MakeDesc, index_t D0, index_t D1>
static void check2d(hk::Result& R, const char* what, std::function<int(int, int)> ref) {
  hk::GuardedBuffer<int> out(D0 * D1);
  out.fill(-1);
  dump2d<MakeDesc, D0, D1><<<1, 1>>>(out.get());
  std::vector<int> e(D0 * D1);
  for (int i = 0; i < D0; ++i) for (int j = 0; j < D1; ++j) e[i * D1 + j] = ref(i, j);
  if (hk::check_equal(R, out.download(), e, what, D1)) out.guards_intact(R);
}

#define MAKER(name, fn) struct name { CK_TILE_HOST_DEVICE constexpr auto operator()() const { return fn; } }
MAKER(RowMajor4x8, (make_row_major<4, 8>()));
MAKER(RowMajor5x3, (make_row_major<5, 3>()));
MAKER(ColMajor4x8, (make_col_major<4, 8>()));
MAKER(ColMajor5x3, (make_col_major<5, 3>()));
MAKER(Transposed4x8, (make_transposed_view<4, 8>()));
MAKER(Transposed6x10, (make_transposed_view<6, 10>()));
MAKER(Xor8x8, (make_xor_swizzled<8, 8>()));
MAKER(Xor16x4, (make_xor_swizzled<16, 4>()));
MAKER(Flat4x8, (make_flattened<4, 8>()));
MAKER(Tiled, (make_tiled<8, 12, 4, 3>()));

TEST_CASE(row_major, 1) {
  check2d<RowMajor4x8, 4, 8>(R, "row_major<4,8>", [](int i, int j) { return i * 8 + j; });
  if (R.passed) check2d<RowMajor5x3, 5, 3>(R, "row_major<5,3>", [](int i, int j) { return i * 3 + j; });
}
TEST_CASE(col_major, 1) {
  check2d<ColMajor4x8, 4, 8>(R, "col_major<4,8>", [](int i, int j) { return j * 4 + i; });
  if (R.passed) check2d<ColMajor5x3, 5, 3>(R, "col_major<5,3>", [](int i, int j) { return j * 5 + i; });
}
TEST_CASE(transposed_view, 2) {
  // 上层坐标 (j, i)：第 0 维长度 N，第 1 维长度 M
  check2d<Transposed4x8, 8, 4>(R, "transposed_view<4,8>（上层 8×4）", [](int j, int i) { return i * 8 + j; });
  if (R.passed) check2d<Transposed6x10, 10, 6>(R, "transposed_view<6,10>（上层 10×6）", [](int j, int i) { return i * 10 + j; });
}
TEST_CASE(xor_swizzled, 2) {
  check2d<Xor8x8, 8, 8>(R, "xor_swizzled<8,8>", [](int i, int j) { return i * 8 + (j ^ (i % 8)); });
  if (R.passed) check2d<Xor16x4, 16, 4>(R, "xor_swizzled<16,4>", [](int i, int j) { return i * 4 + (j ^ (i % 4)); });
}

template <class MakeDesc>
__global__ void dump_flat(int* out, int n) {
  constexpr auto d = MakeDesc{}();
  for (int k = 0; k < n; ++k) out[k] = d.calculate_offset(make_tuple(k));
  out[n] = d.get_num_of_top_dimension();
}
TEST_CASE(flattened, 2) {
  hk::GuardedBuffer<int> out(33);
  out.fill(-1);
  dump_flat<Flat4x8><<<1, 1>>>(out.get(), 32);
  std::vector<int> e(33);
  for (int k = 0; k < 32; ++k) e[k] = k;
  e[32] = 1;  // 上层只有 1 个维度
  if (hk::check_equal(R, out.download(), e, "flattened<4,8>（最后一项为上层维度数，应为 1）")) out.guards_intact(R);
}

template <class MakeDesc>
__global__ void dump_tiled(int* out) {
  constexpr auto d = MakeDesc{}();
  int p = 0;
  for (int I = 0; I < 2; ++I) for (int J = 0; J < 4; ++J) for (int ii = 0; ii < 4; ++ii) for (int jj = 0; jj < 3; ++jj)
    out[p++] = d.calculate_offset(make_tuple(I, J, ii, jj));
  out[p] = d.get_num_of_top_dimension();
}
TEST_CASE(tiled, 3) {
  hk::GuardedBuffer<int> out(97);
  out.fill(-1);
  dump_tiled<Tiled><<<1, 1>>>(out.get());
  std::vector<int> e;
  for (int I = 0; I < 2; ++I) for (int J = 0; J < 4; ++J) for (int ii = 0; ii < 4; ++ii) for (int jj = 0; jj < 3; ++jj)
    e.push_back((I * 4 + ii) * 12 + J * 3 + jj);
  e.push_back(4);
  if (hk::check_equal(R, out.download(), e, "tiled<8,12,4,3>，按 (I,J,ii,jj) 字典序排列（最后一项为上层维度数 4）")) out.guards_intact(R);
}
HK_MAIN()
```

- [ ] **Step 2: 参考答案 `solutions/lab3-cktile-concepts/a1-layout/layout.hip`**

```cpp
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_row_major() {
  return make_naive_tensor_descriptor(make_tuple(number<M>{}, number<N>{}), make_tuple(number<N>{}, number<1>{}));
}

template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_col_major() {
  return make_naive_tensor_descriptor(make_tuple(number<M>{}, number<N>{}), make_tuple(number<1>{}, number<M>{}));
}

template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_transposed_view() {
  // 下层维度 0 (i) → 上层维度 1；下层维度 1 (j) → 上层维度 0
  return transform_tensor_descriptor(make_row_major<M, N>(),
      make_tuple(make_pass_through_transform(number<M>{}), make_pass_through_transform(number<N>{})),
      make_tuple(sequence<0>{}, sequence<1>{}),
      make_tuple(sequence<1>{}, sequence<0>{}));
}

template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_flattened() {
  return transform_tensor_descriptor(make_row_major<M, N>(),
      make_tuple(make_merge_transform(make_tuple(number<M>{}, number<N>{}))),
      make_tuple(sequence<0, 1>{}),
      make_tuple(sequence<0>{}));
}

template <index_t M, index_t N, index_t TM, index_t TN>
CK_TILE_HOST_DEVICE constexpr auto make_tiled() {
  // i = I*TM + ii 拆成 (I, ii)，j = J*TN + jj 拆成 (J, jj)，上层顺序 (I, J, ii, jj)
  return transform_tensor_descriptor(make_row_major<M, N>(),
      make_tuple(make_unmerge_transform(make_tuple(number<M / TM>{}, number<TM>{})),
                 make_unmerge_transform(make_tuple(number<N / TN>{}, number<TN>{}))),
      make_tuple(sequence<0>{}, sequence<1>{}),
      make_tuple(sequence<0, 2>{}, sequence<1, 3>{}));
}

template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzled() {
  return transform_tensor_descriptor(make_row_major<M, N>(),
      make_tuple(make_xor_transform(make_tuple(number<M>{}, number<N>{}))),
      make_tuple(sequence<0, 1>{}),
      make_tuple(sequence<0, 1>{}));
}
```

- [ ] **Step 3: 学生骨架（难度 1 → 2）**

`make_row_major` 完整给出（作为示例）。其余函数体统一占位 `return make_row_major<M, N>();`（`make_tiled` 的占位为 4 维的错误 naive descriptor `make_naive_tensor_descriptor(make_tuple(number<M/TM>{}, number<N/TN>{}, number<TM>{}, number<TN>{}), make_tuple(number<0>{}, number<0>{}, number<0>{}, number<0>{}))`，`make_flattened` 的占位为 1 维 naive descriptor `make_naive_tensor_descriptor(make_tuple(number<M*N>{}), make_tuple(number<0>{}))`，保证测试中的 `make_tuple(k)` / 4 维坐标能编译），每个函数前：
- `// TODO(a1-1) col_major：只改 stride（讲义 §1.2）`
- `// TODO(a1-2) transposed_view：用两个 pass_through 变换重排维度。transform_tensor_descriptor 的第 3、4 个参数分别是"每个变换吃哪些下层维度"和"产出哪些上层维度"（讲义 §1.3）`
- `// TODO(a1-3) flattened：merge 把多个下层维度合成一个上层维度（讲义 §1.4）`
- `// TODO(a1-4) tiled：unmerge 把一个下层维度拆成多个上层维度；想清楚上层维度编号 0..3 分别是谁（讲义 §1.5）`
- `// TODO(a1-5) xor_swizzled：xor 变换（讲义 §1.6；读 coordinate_transform.hpp 中 struct xor_t 的 calculate_lower_index）`

骨架中的占位必须实测可编译，且原样提交时 row_major 通过、其余全部 FAIL。

- [ ] **Step 4: mutants**

- `m1_colmajor_swapped.hip`：col_major 的 stride 写成 `(M, 1)` → `col_major` FAIL。
- `m2_transpose_wrong_dims.hip`：transposed_view 的上层维度写成 `sequence<0>{}, sequence<1>{}`（没有交换）→ 上层形状仍为 M×N，`transposed_view` FAIL（按 N×M 坐标访问时偏移错误）。
- `m3_tiled_wrong_order.hip`：tiled 上层维度写成 `sequence<0, 1>{}, sequence<2, 3>{}`（得到 (I, ii, J, jj)）→ `tiled` FAIL。
- `m4_xor_dims.hip`：row-major 底座换成 col-major 再做 xor → `xor_swizzled` FAIL。

- [ ] **Step 5: README**：书面问题：
1. `transposed_view` 没有移动任何数据，它和 Lab 1 a2 中的转置有什么本质区别？什么时候"换视图"就够了，什么时候必须真的搬数据？
2. 用 xor_swizzled<8,8> 的偏移表说明：按列读取时，8 个元素落在多少个不同的"bank"上（假设 8 个 bank，bank = offset % 8）？与 row-major 比较。
3. `make_tiled` 得到的 4 维视图如何用来表达"block (I,J) 负责的 tile 中第 (ii,jj) 个元素"？这与 Lab 1 a4 的手写下标计算对应关系是什么？

- [ ] **Step 6: 验证与 Commit**

Run: `cd lab3-cktile-concepts && make verify` → Expected: `VERIFY OK`。
```bash
git add lab3-cktile-concepts/Makefile lab3-cktile-concepts/a1-layout solutions/lab3-cktile-concepts/a1-layout
git commit -m "feat(lab3): a1 tensor descriptors and coordinate transforms"
```

---
### Task 15: Lab 3 · a2 用 CK-Tile 做 LDS 转置

**学习目标：** 2D tile distribution（多个 H 段、Y 维分布在两个 X 维上）；LDS 上的 tensor view（`address_space_enum::lds`）；**同一块 LDS 用两个不同 descriptor 读写**（a1 的转置视图 + padding）；`block_sync_lds()`；`pad_tensor_view` 处理 2D 边界（实测：不加 pad 时 33×31 有 806 个元素错误）。与 Lab 1 a2 做同一件事，对照"手写下标"与"layout 描述"。

**Files:**
- Create: `lab3-cktile-concepts/a2-tile-transpose/{tile_transpose.hip,test.cpp,README.md}`
- Create: `solutions/lab3-cktile-concepts/a2-tile-transpose/tile_transpose.hip` + `mutants/{m1_no_pad.hip,m2_lds_read_same_layout.hip,m3_out_origin_swapped.hip,m4_one_wave_block.hip}`

**Interfaces:**
```cpp
// in: rows×cols 行主序；out: cols×rows 行主序
void launch_ck_transpose(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab3-cktile-concepts/a2-tile-transpose/test.cpp`**

与 Task 6 的 test.cpp 结构相同，完整代码：
```cpp
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

constexpr double kCkTransposeRatio = 0.50;  // Step 5 校准后写死
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
  if (R.metric < kCkTransposeRatio * copy)
    hk::fail(R, "带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）", R.metric,
             kCkTransposeRatio * copy, kCkTransposeRatio * 100, copy);
}
HK_MAIN()
```

- [ ] **Step 2: 参考答案 `solutions/lab3-cktile-concepts/a2-tile-transpose/tile_transpose.hip`**（实测正确的版本）

```cpp
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

constexpr index_t TM = 64, TN = 64;  // 每个 block 处理 64×64 的 tile，256 线程

// 64×64 tile 的分布：每线程一次读同一行 4 个连续元素；一行 64 元素 = 16 个线程；
// 256 线程一次覆盖 16 行，重复 4 次覆盖 64 行。
//   X0（行，64）= H<4(Y0 重复), 4(P0 warp), 4(P1 lane 高位)>
//   X1（列，64）= H<16(P1 lane 低位), 4(Y1 向量)>
CK_TILE_DEVICE constexpr auto make_tile_dist() {
  return make_static_tile_distribution(tile_distribution_encoding<
      sequence<>,
      tuple<sequence<4, 4, 4>, sequence<16, 4>>,
      tuple<sequence<1>, sequence<1, 2>>,   // P0 → X0 段 1；P1 → X0 段 2 与 X1 段 0
      tuple<sequence<1>, sequence<2, 0>>,
      sequence<1, 2>,                       // Y0 → X0 段 0；Y1 → X1 段 1
      sequence<0, 1>>{});
}

__global__ void ck_transpose_kernel(const float* in, float* out, index_t rows, index_t cols) {
  __shared__ float smem[TM * (TN + 1)];  // +1 padding
  // 全局视图 + pad：tile 超出矩阵的部分读到 0、写被丢弃（逐维检查，而不只是检查线性长度）
  auto vin = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(in, make_tuple(rows, cols), make_tuple(cols, 1), number<1>{}, number<1>{}),
      make_tuple(number<TM>{}, number<TN>{}), sequence<true, true>{});
  auto vout = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(out, make_tuple(cols, rows), make_tuple(rows, 1), number<1>{}, number<1>{}),
      make_tuple(number<TN>{}, number<TM>{}), sequence<true, true>{});
  // 同一块 LDS 的两个视图：写入用 (TM, TN) 行主序（行 stride = TN+1）；读出用 (TN, TM) 的转置视图
  auto lds_w = make_naive_tensor_view<address_space_enum::lds>(
      smem, make_tuple(number<TM>{}, number<TN>{}), make_tuple(number<TN + 1>{}, number<1>{}), number<1>{}, number<1>{});
  auto lds_r = make_naive_tensor_view<address_space_enum::lds>(
      smem, make_tuple(number<TN>{}, number<TM>{}), make_tuple(number<1>{}, number<TN + 1>{}), number<1>{}, number<1>{});

  constexpr auto dist = make_tile_dist();
  const index_t r0 = static_cast<index_t>(blockIdx.y) * TM, c0 = static_cast<index_t>(blockIdx.x) * TN;

  auto t = load_tile(make_tile_window(vin, make_tuple(number<TM>{}, number<TN>{}), {r0, c0}, dist));
  auto ww = make_tile_window(lds_w, make_tuple(number<TM>{}, number<TN>{}), {0, 0}, dist);
  store_tile(ww, t);
  block_sync_lds();
  auto t2 = load_tile(make_tile_window(lds_r, make_tuple(number<TN>{}, number<TM>{}), {0, 0}, dist));
  auto ow = make_tile_window(vout, make_tuple(number<TN>{}, number<TM>{}), {c0, r0}, dist);
  store_tile(ow, t2);
}

void launch_ck_transpose(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 grid((cols + TN - 1) / TN, (rows + TM - 1) / TM);
  ck_transpose_kernel<<<grid, 256, 0, stream>>>(in, out, rows, cols);
}
```

- [ ] **Step 3: 学生骨架（难度 2）**

给出：`TM/TN`、`smem` 声明、`vin`（含 pad，作为示例）、`launch` 函数。挖空：
- `// TODO(a2-1): 仿照 vin 构造 vout（注意 out 的形状是 cols×rows）（讲义 §2.3）`
- `// TODO(a2-2): 设计 64×64 tile 的 distribution（讲义 §2.1–2.2）。要求：256 线程；同一 wavefront 内的线程读同一行的连续元素。先在纸上写出 H/P/Y 的划分，再翻译成 encoding。`
- `// TODO(a2-3): 为 smem 构造两个 LDS 视图：写入视图与转置读出视图（复习 a1 的 transposed_view 思想，这里用 stride 直接表达）（讲义 §2.4）`
- `// TODO(a2-4): global → 寄存器 → LDS → 同步 → LDS（转置视图）→ 寄存器 → global（讲义 §2.5）`
骨架 kernel 中除 `vin` 外无其他语句（`(void)vin;`），原样提交时全部 FAIL。

- [ ] **Step 4: mutants**

- `m1_no_pad.hip`：去掉两处 `pad_tensor_view`（直接用 naive view）→ `ragged` 中 33×31 FAIL（实测 806 个错误元素）。
- `m2_lds_read_same_layout.hip`：读出也用 `lds_w`（不转置）→ 输出变成 tile 内未转置 → FAIL。
- `m3_out_origin_swapped.hip`：输出窗口原点写成 `{r0, c0}` → `multi_tile` FAIL。
- `m4_one_wave_block.hip`：block 64 线程、encoding 改为 X0 = H<16(Y0), 4(P1 lane 高位)>、X1 = H<16(P1 lane 低位), 4(Y1)>，P 只有 lane 一维 → 正确但每 block 只有 1 个 wave（候选慢 mutant，依赖校准；若不够慢则从 mutants 删除并按兜底规则把阈值设为 S_min/1.5，记录在审查报告中）。

- [ ] **Step 5: 校准 `kCkTransposeRatio`**：同 Task 6 Step 6（solution 对 m4）。另外记录 Lab 1 a2 参考答案的带宽，写入 README 书面问题 3 的对比。

- [ ] **Step 6: README**：Hint 1：先确定"一个线程一次读几个元素、一行需要几个线程、256 线程覆盖几行、需要重复几次"；Hint 2：lane id 同时参与行和列的划分时，P1 对应两个 H 段（`sequence<1, 2>` / `sequence<2, 0>`）。书面问题：
1. 去掉 `pad_tensor_view` 时，为什么 1×1 的矩阵结果碰巧正确，而 33×31 错误？（提示：buffer 指令只检查线性偏移）
2. `lds_r` 的 stride 为 `(1, TN+1)`，它和 a1 的 `make_transposed_view` 有什么关系？
3. 比较本作业与 Lab 1 a2 参考答案的带宽，解释差异。

- [ ] **Step 7: 验证与 Commit**

Run: `cd lab3-cktile-concepts && make verify` → Expected: `VERIFY OK`。
```bash
git add lab3-cktile-concepts/a2-tile-transpose solutions/lab3-cktile-concepts/a2-tile-transpose
git commit -m "feat(lab3): a2 CK-Tile LDS transpose with padded views"
```

---
### Task 16: Lab 3 · a3 行归约（CK-Tile 收尾，难度 3）

**学习目标：** 在 tile 上做归约：`block_tile_reduce`（线程内归约 Y 维）+ `block_tile_reduce_sync`（跨 lane 归约）；`move_tile_window` 沿列方向滑动窗口；归约结果的分布（行方向分布、列方向被"复制"）；**`guaranteed_last_dim_vector_length` 必须如实声明**（实测：cols 不是 4 的倍数时声明 4 会算错）；**运行时分派到不同编译期配置**（cols % 4 == 0 时走向量化版本）——这是 CK-Tile 实际 kernel 中的常见模式。限制（来自源码注释）：`block_tile_reduce` 要求至少有一个 Y 维不被归约；`block_tile_reduce_sync` 不支持跨 warp 归约。以下参考答案已实测（含 33×31、1000×4099）。

**Files:**
- Create: `lab3-cktile-concepts/a3-row-sum/{row_sum.hip,test.cpp,README.md}`
- Create: `solutions/lab3-cktile-concepts/a3-row-sum/row_sum.hip` + `mutants/{m1_no_sync.hip,m2_no_window_move.hip,m3_vec4_always.hip,m4_vec1_always.hip}`

**Interfaces:**
```cpp
// out[r] = Σ_c in[r*cols + c]，in 为 rows×cols 行主序 fp32
void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream);
```

- [ ] **Step 1: 测试 `lab3-cktile-concepts/a3-row-sum/test.cpp`**

```cpp
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

constexpr double kRowSumRatio = 0.50;  // Step 5 校准后写死
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
  if (R.metric < kRowSumRatio * copy)
    hk::fail(R, "读带宽 %.0f GB/s，低于目标 %.0f GB/s（= %.0f%% × 本次实测拷贝带宽 %.0f GB/s）。cols 是 4 的倍数时是否用了向量化配置？",
             R.metric, kRowSumRatio * copy, kRowSumRatio * 100, copy);
}
HK_MAIN()
```

- [ ] **Step 2: 参考答案 `solutions/lab3-cktile-concepts/a3-row-sum/row_sum.hip`**

```cpp
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
using namespace ck_tile;

constexpr index_t kWarps = 4, kRep = 2, kLanes = 64;
constexpr index_t TM = kRep * kWarps;  // 每个 block 处理 8 行：每个 warp 负责 2 行（Y0 重复 2 次）

// 每个 warp 的 64 个 lane 沿列方向排开，每 lane 读 kVec 个连续元素；TN = 64 * kVec 列为一个 tile。
// 行方向只映射到 warp 与 Y0 —— 因此列方向的归约只需跨 lane（block_tile_reduce_sync 不支持跨 warp）。
template <index_t kVec>
__global__ void row_sum_kernel(const float* in, float* out, index_t rows, index_t cols) {
  constexpr index_t TN = kLanes * kVec;
  // kVec 同时作为"保证的最内维向量长度"：只有 cols % kVec == 0 时才能如实声明
  auto v = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(in, make_tuple(rows, cols), make_tuple(cols, 1), number<kVec>{}, number<1>{}),
      make_tuple(number<TM>{}, number<TN>{}), sequence<true, true>{});
  constexpr auto dist = make_static_tile_distribution(tile_distribution_encoding<
      sequence<>,
      tuple<sequence<kRep, kWarps>, sequence<kLanes, kVec>>,
      tuple<sequence<1>, sequence<2>>,   // P0(warp) → 行段 1；P1(lane) → 列段 0
      tuple<sequence<1>, sequence<0>>,
      sequence<1, 2>,                    // Y0 → 行段 0（不被归约，满足 block_tile_reduce 的要求）；Y1 → 列段 1
      sequence<0, 1>>{});

  const index_t r0 = static_cast<index_t>(blockIdx.x) * TM;
  auto win = make_tile_window(v, make_tuple(number<TM>{}, number<TN>{}), {r0, 0}, dist);
  auto add = [](float a, float b) { return a + b; };

  auto acc = block_tile_reduce<float>(load_tile(win), sequence<1>{}, add, 0.0f);  // 线程内：归约 Y1
  for (index_t c = TN; c < cols; c += TN) {
    move_tile_window(win, {0, TN});
    block_tile_reduce(acc, load_tile(win), sequence<1>{}, add);
  }
  block_tile_reduce_sync(acc, add);  // 跨 lane 归约，并把结果广播给同一行的所有 lane

  auto vo = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(out, make_tuple(rows), make_tuple(1), number<1>{}, number<1>{}),
      make_tuple(number<TM>{}), sequence<true>{});
  auto ow = make_tile_window(vo, make_tuple(number<TM>{}), {r0}, acc.get_tile_distribution());
  store_tile(ow, acc);
}

void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  const int grid = (rows + TM - 1) / TM;
  if (cols % 4 == 0) row_sum_kernel<4><<<grid, kWarps * kLanes, 0, stream>>>(in, out, rows, cols);  // 运行时分派
  else               row_sum_kernel<1><<<grid, kWarps * kLanes, 0, stream>>>(in, out, rows, cols);
}
```

- [ ] **Step 3: 学生骨架（难度 3）**

只给 include、`launch_ck_row_sum` 签名与空函数体，以及设计引导注释：
```cpp
// TODO(a3-1): 自主设计 CK-Tile 行归约（讲义 §3.1–3.5）
//   需要决定：一个 block 处理几行、行/列分别映射到 warp/lane/Y 的哪一段、tile 宽度、怎样沿列滑动窗口。
//   必须遵守的约束（读 ck_tile/ops/reduce/block/block_reduce.hpp 开头的注释）：
//     · block_tile_reduce 要求至少一个 Y 维不被归约
//     · block_tile_reduce_sync 只做 warp 内（跨 lane）归约 —— 所以一行不能跨 warp
//   边界：行数、列数都可能不是 tile 的整数倍（复习 a2 的 pad_tensor_view）。
//   性能：bandwidth_aligned 用例中 cols 是 4 的倍数；想想 make_naive_tensor_view 的倒数第二个参数什么时候可以写 4。
void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  // >>> 你的代码 <<<
}
```

- [ ] **Step 4: mutants**

- `m1_no_sync.hip`：删去 `block_tile_reduce_sync` → 每个 lane 只持有部分和，FAIL。
- `m2_no_window_move.hip`：删去 for 循环（只归约第一个 tile）→ `many_col_tiles` FAIL。
- `m3_vec4_always.hip`：总是 `row_sum_kernel<4>` → `ragged_cols`（cols=31/4099）FAIL（实测会算错）。
- `m4_vec1_always.hip`：总是 `row_sum_kernel<1>` → 正确但慢，`bandwidth_aligned` FAIL（依赖校准；不够慢则按兜底规则处理并记录）。

- [ ] **Step 5: 校准 `kRowSumRatio`**：同 Task 6 Step 6（solution 对 m4）。

- [ ] **Step 6: README**：Hint 1：从参考设计空间出发——"一行只由一个 warp 处理"；画出 8×256 tile 中 (warp, lane, y0, y1) 与 (行, 列) 的对应关系。Hint 2：`block_tile_reduce<float>(tile, sequence<1>{}, f, 0.0f)` 返回的张量少了第 1 维；循环中用 4 参数重载累加；最后 sync；输出窗口的 distribution 用 `acc.get_tile_distribution()`。书面问题：
1. 为什么 cols % 4 ≠ 0 时声明向量长度 4 会出错？（提示：每行起始地址的对齐）
2. `block_tile_reduce_sync` 之后，同一行的 64 个 lane 都持有结果，`store_tile` 时会重复写吗？读 `store_tile` 相关源码说明 CK-Tile 如何处理复制（R）维/被归约维。
3. 如果要支持一行跨 warp（例如 cols 很大而 rows 很小），需要额外做什么？（铺垫：跨 warp 归约要经过 LDS，类似 Lab 1 a3）
4. 设计说明（3–5 句话）。

- [ ] **Step 7: 验证与 Commit**

Run: `cd lab3-cktile-concepts && make verify` → Expected: `VERIFY OK`（a1–a3）。
```bash
git add lab3-cktile-concepts/a3-row-sum solutions/lab3-cktile-concepts/a3-row-sum
git commit -m "feat(lab3): a3 CK-Tile row reduction with runtime vector dispatch"
```

---
### Task 17: Lab 3 讲义 + 审查

**Files:**
- Create: `lab3-cktile-concepts/tutorial.html`, `lab3-cktile-concepts/answers.md`, `docs/reviews/lab3.md`

章节（小节编号须与 a1–a3 的 TODO 引用一致）：
1. **Layout = 坐标变换的组合**：§1.1 tensor descriptor 的上层/下层维度与 `calculate_offset`；§1.2 naive descriptor 与 stride（行主序/列主序）；§1.3 pass_through 与维度重排（transform_tensor_descriptor 四个参数逐一讲解，配 SVG 数据流图）；§1.4 merge；§1.5 unmerge 与分块视图；§1.6 xor swizzle 与 LDS bank（引用 `coordinate_transform.hpp` 中 `xor_t::calculate_lower_index`）。**widget（核心）：layout explorer**——选择变换链（row/col-major、transpose、merge、tiled、xor）与尺寸，显示网格：每个逻辑坐标格子里标出线性偏移，并用颜色显示 bank（offset % 32）。→ assignment-card a1
2. **2D tile distribution 与 LDS**：§2.1 2D encoding 的读法（R/H/P/Y 表格 + 线程-元素映射图）；§2.2 一个 P 维映射到多个 H 段（lane 同时覆盖行和列）；**widget：distribution explorer**——输入 H 段与 P/Y 映射（预设几组），画出 tile 中每个元素属于哪个 (warp, lane, y)，悬停高亮同一线程的所有元素；§2.3 `pad_tensor_view` 与 2D 边界（为什么线性长度检查不够，图示 33×31 的越界列落到下一行）；§2.4 LDS 视图、padding、同一块内存两个视图；§2.5 `block_sync_lds` 与完整数据流。→ assignment-card a2
3. **在 tile 上归约**：§3.1 `block_tile_reduce` 与归约后的分布；§3.2 `block_tile_reduce_sync` 与 lane 间 xor shuffle（与 Lab 1 a3 的 `__shfl_down` 对照）；§3.3 `move_tile_window`；§3.4 guaranteed vector length 与对齐；§3.5 运行时分派到编译期配置。→ assignment-card a3
4. **回顾与展望**：用 CK-Tile 概念重新描述 Lab 1 a4 的分块 GEMM；预告 Lab 4（warp GEMM / MFMA 的 distribution）。

- [ ] **Step 1: 编写讲义与 2 个 widget**（要求同 Task 9 Step 1–2；CK-Tile 片段逐一抽出编译）
- [ ] **Step 2: `answers.md`**
- [ ] **Step 3: 浏览器验证与小节编号交叉核对**（同 Task 9 Step 4–5）
- [ ] **Step 4: 用 `docs/reviews/PROMPTS.md` 派发 R1/R2/R3（`{LAB}=lab3-cktile-concepts`），处理结果，写 `docs/reviews/lab3.md`，必要时复查**（流程同 Task 10 Step 3–5）
- [ ] **Step 5: Commit**

```bash
git add lab3-cktile-concepts solutions/lab3-cktile-concepts docs/reviews/lab3.md
git commit -m "docs(lab3): CK-Tile core concepts tutorial, reviewed"
```

---

### Task 18: 收尾：README、总目录、全量验证、整体审查

**Files:**
- Modify: `README.md`, `site/index.html`
- Create: `Makefile`（根目录）, `docs/reviews/phase1-summary.md`

- [ ] **Step 1: 根目录 `Makefile`**

```make
LABS := lab0-env lab1-hip-basics lab2-cktile-intro lab3-cktile-concepts
.PHONY: verify-all test-infra
test-infra:
	python3 common/test_grade.py
verify-all: test-infra
	@set -e; for l in $(LABS); do echo "######## $$l"; $(MAKE) -s -C $$l verify; done
```

- [ ] **Step 2: `README.md`**（中文）：课程简介与路线图（Lab 0–8，标注已完成的第一期）；快速开始（`cd lab0-env && make grade`；用浏览器打开 `site/index.html`）；每个 Lab 的学习流程（读讲义 → 做 aK → `make grade A=aK` → 写 answers.md）；目录结构说明；"关于 solutions/：卡住超过 1 小时再看"；共享 GPU 说明（自动选卡、`HIP_VISIBLE_DEVICES` 覆盖）；给维护者：`make verify-all`。

- [ ] **Step 3: `site/index.html`**：确认 Lab 0–3 卡片链接全部可点（Playwright 逐个点击，页面 title 正确，无 console error）。

- [ ] **Step 4: 全量验证**

Run: `make verify-all`
Expected: `test_grade.py` 全部 ok；4 个 Lab 均 `VERIFY OK`。把输出摘要（每个性能用例的实测值）写入 `docs/reviews/phase1-summary.md`。

- [ ] **Step 5: 整体连续性审查**：派发 1 个 subagent（学生视角，禁止看 solutions），从 `site/index.html` 开始按顺序通读 Lab 0→3 的讲义和所有 README，**只评估跨 Lab 的连续性**：每个 Assignment 用到的概念是否都在之前讲过；Lab 之间有无断层；术语是否一致（wavefront/warp、tile/块等）。报告写入 `docs/reviews/phase1-summary.md`，属实的问题修复后重新 `make verify-all`。

- [ ] **Step 6: Commit**

```bash
git add README.md Makefile site/index.html docs/reviews/phase1-summary.md
git commit -m "docs: phase 1 README, course index and verification summary"
```

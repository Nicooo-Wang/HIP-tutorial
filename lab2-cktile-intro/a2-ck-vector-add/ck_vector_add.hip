// Lab 2 · a2：用 CK-Tile 写 vector add（c[i] = a[i] + b[i]）
// 与 Lab 1 a1 做同一件事，但这次不手写下标计算，而是用 CK-Tile 的“四件套”描述：
//   tensor view（有形状的全局内存视图）→ tile distribution（tile 内每个元素归哪个线程）
//   → tile window（视图上的一个窗口）→ load_tile / sweep_tile / store_tile
// 判分：在 lab2-cktile-intro/ 目录下运行 make grade A=a2；调试：make run A=a2 CASE=用例名
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

constexpr index_t kWarps = 4;               // 每个 block 4 个 wavefront
constexpr index_t kLanes = 64;              // 每个 wavefront 64 个 lane
constexpr index_t kVec = 4;                 // 每个线程连续处理 4 个 float（16 字节向量访存）
constexpr index_t kBlock = kWarps * kLanes; // 256 线程
constexpr index_t kTile = kBlock * kVec;    // 每个 block 处理 1024 个元素

__global__ void ck_vector_add_kernel(const float* a, const float* b, float* c, index_t n) {
  // 1) tensor view：给裸指针加上形状 (n)、stride (1)。这是 a 的完整示例：
  //    越界访问由 buffer 指令按长度 n 屏蔽——所以后面不需要 if (i < n)。
  //    最后两个参数是“向量长度 kVec”和“最后一维 stride 为 1”的编译期承诺；不写的话 CK-Tile 一次只访问 1 个元素（讲义 §3.1）。
  auto va = make_naive_tensor_view<address_space_enum::global>(a, make_tuple(n), make_tuple(1), number<kVec>{}, number<1>{});
  (void)va;  // 写完后面的代码后，这一行可以删掉

  // TODO(a2-1): 仿照 va 构造 vb、vc（讲义 §3.1）
  // >>> 你的代码 <<<

  // TODO(a2-2): 写出 tile_distribution_encoding 的 6 个模板参数并 make_static_tile_distribution（讲义 §3.2）。
  //   先在纸上画出 kTile 个元素如何分给 kWarps×kLanes 个线程。
  // >>> 你的代码 <<<

  // TODO(a2-3): 计算本 block 的窗口原点并创建 3 个 tile window（讲义 §3.3）。
  //   注意 blockIdx.x 是 unsigned，需要 static_cast<index_t>
  // >>> 你的代码 <<<

  // TODO(a2-4): load、sweep_tile 中逐元素相加、store（讲义 §3.4）
  // >>> 你的代码 <<<
}

void launch_ck_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  const int grid = (n + kTile - 1) / kTile;  // 每个 block 处理 kTile 个元素，向上取整
  ck_vector_add_kernel<<<grid, kBlock, 0, stream>>>(a, b, c, n);
}

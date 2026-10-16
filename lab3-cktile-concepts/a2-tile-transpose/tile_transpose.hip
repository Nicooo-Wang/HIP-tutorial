// Lab 3 · a2：用 CK-Tile 做 LDS 转置
// 与 Lab 1 a2 做同一件事：out（cols×rows）= in（rows×cols）的转置。
// 这次不手写 (threadIdx, 下标) 公式，而是用 CK-Tile 描述：
//   两个全局视图（带 pad）+ 一个 2D tile distribution + 同一块 LDS 上的两个视图（写入视图 / 转置读出视图）
// 判分：在 lab3-cktile-concepts/ 目录下运行 make grade A=a2；调试：make run A=a2 CASE=用例名
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

constexpr index_t TM = 64, TN = 64;  // 每个 block 处理 64×64 的 tile，256 线程（4 个 wavefront）

__global__ void ck_transpose_kernel(const float* in, float* out, index_t rows, index_t cols) {
  __shared__ float smem[TM * (TN + 1)];  // 每行多留 1 个 float 的 padding，避免 LDS bank 冲突

  // 全局输入视图：形状 (rows, cols)，并用 pad_tensor_view 让 tile 超出矩阵的部分读到 0、写被丢弃。
  // naive 视图的越界屏蔽只按“线性总长度”，2D tile 跨过行/列边界时不会被拦（讲义 §2.3）。
  auto vin = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(in, make_tuple(rows, cols), make_tuple(cols, 1), number<1>{}, number<1>{}),
      make_tuple(number<TM>{}, number<TN>{}), sequence<true, true>{});
  (void)vin;  // 写完后面的代码后，这一行可以删掉

  // TODO(a2-1): 仿照 vin 构造 vout（注意 out 的形状是 cols×rows）（讲义 §2.3）
  // >>> 你的代码 <<<

  // TODO(a2-2): 设计 64×64 tile 的 distribution（讲义 §2.1–2.2）。要求：256 线程；同一 wavefront 内的线程读同一行的连续元素。
  //   先在纸上写出 H/P/Y 的划分，再翻译成 encoding。
  // >>> 你的代码 <<<

  // TODO(a2-3): 为 smem 构造两个 LDS 视图：写入视图与转置读出视图（复习 a1 的 transposed_view 思想，这里用 stride 直接表达）（讲义 §2.4）
  // >>> 你的代码 <<<

  // TODO(a2-4): global → 寄存器 → LDS → 同步 → LDS（转置视图）→ 寄存器 → global（讲义 §2.5）
  //   本 block 负责输入的 (blockIdx.y * TM, blockIdx.x * TN) 处的 tile，注意 blockIdx 是 unsigned
  // >>> 你的代码 <<<
}

void launch_ck_transpose(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 grid((cols + TN - 1) / TN, (rows + TM - 1) / TM);
  ck_transpose_kernel<<<grid, 256, 0, stream>>>(in, out, rows, cols);
}

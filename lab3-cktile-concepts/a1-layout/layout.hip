// Lab 3 · a1：layout —— tensor descriptor 与坐标变换
// tensor descriptor = 一串坐标变换：把“上层逻辑坐标”映射成“线性偏移”。
// 同一块内存可以有不同的视图（转置、展平、分块、swizzle），而且都不搬数据。
// 本文件里的函数会在设备端被调用，测试在所有上层坐标上取 calculate_offset 并与 CPU 公式对比。
// 判分：在 lab3-cktile-concepts/ 目录下运行 make grade A=a1；调试：make run A=a1 CASE=用例名
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
using namespace ck_tile;

// 示例（已给出）：M×N 行主序，(i, j) → i*N + j。
// naive descriptor 由“各维长度”和“各维 stride”给出。
template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_row_major() {
  return make_naive_tensor_descriptor(make_tuple(number<M>{}, number<N>{}), make_tuple(number<N>{}, number<1>{}));
}

// TODO(a1-1) col_major：只改 stride（讲义 §1.2）。M×N 列主序，(i, j) → j*M + i。
template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_col_major() {
  // >>> 你的代码 <<<
  return make_row_major<M, N>();  // 占位：形状对，但 stride 不对，请替换
}

// TODO(a1-2) transposed_view：用两个 pass_through 变换重排维度。transform_tensor_descriptor 的第 3、4 个参数分别是“每个变换吃哪些下层维度”和“产出哪些上层维度”（讲义 §1.3）
//   基于 M×N 行主序的底座，上层坐标是 (j, i)（j < N, i < M），偏移为 i*N + j。
template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_transposed_view() {
  // >>> 你的代码 <<<
  return make_row_major<M, N>();  // 占位：没有做任何变换，请替换
}

// TODO(a1-3) flattened：merge 把多个下层维度合成一个上层维度（讲义 §1.4）
//   基于 M×N 行主序的底座，上层只有 1 个维度，坐标 k → 偏移 k。
template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_flattened() {
  // >>> 你的代码 <<<
  return make_naive_tensor_descriptor(make_tuple(number<M * N>{}), make_tuple(number<0>{}));  // 占位：1 维但 stride 为 0，请替换
}

// TODO(a1-4) tiled：unmerge 把一个下层维度拆成多个上层维度；想清楚上层维度编号 0..3 分别是谁（讲义 §1.5）
//   基于 M×N 行主序的底座，上层坐标 (I, J, ii, jj) → (I*TM+ii)*N + J*TN + jj。
template <index_t M, index_t N, index_t TM, index_t TN>
CK_TILE_HOST_DEVICE constexpr auto make_tiled() {
  // >>> 你的代码 <<<
  return make_naive_tensor_descriptor(make_tuple(number<M / TM>{}, number<N / TN>{}, number<TM>{}, number<TN>{}),
                                      make_tuple(number<0>{}, number<0>{}, number<0>{}, number<0>{}));  // 占位：4 维但 stride 全 0，请替换
}

// TODO(a1-5) xor_swizzled：xor 变换（讲义 §1.6；读 coordinate_transform.hpp 中 struct xor_t 的 calculate_lower_index）
//   基于 M×N 行主序的底座，上层坐标 (i, j) → i*N + (j ^ (i % N))。
template <index_t M, index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzled() {
  // >>> 你的代码 <<<
  return make_row_major<M, N>();  // 占位：没有 swizzle，请替换
}

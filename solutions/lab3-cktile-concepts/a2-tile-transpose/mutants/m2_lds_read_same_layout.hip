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
  auto t2 = load_tile(make_tile_window(lds_w, make_tuple(number<TM>{}, number<TN>{}), {0, 0}, dist));
  auto ow = make_tile_window(vout, make_tuple(number<TN>{}, number<TM>{}), {c0, r0}, dist);
  store_tile(ow, t2);
}

void launch_ck_transpose(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 grid((cols + TN - 1) / TN, (rows + TM - 1) / TM);
  ck_transpose_kernel<<<grid, 256, 0, stream>>>(in, out, rows, cols);
}

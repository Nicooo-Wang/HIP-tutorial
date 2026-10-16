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
  if (cols % 2 == 0) row_sum_kernel<4><<<grid, kWarps * kLanes, 0, stream>>>(in, out, rows, cols);  // mutant：偶数列就承诺向量 4，cols % 4 == 2 时行尾向量跨进下一行
  else               row_sum_kernel<1><<<grid, kWarps * kLanes, 0, stream>>>(in, out, rows, cols);
}

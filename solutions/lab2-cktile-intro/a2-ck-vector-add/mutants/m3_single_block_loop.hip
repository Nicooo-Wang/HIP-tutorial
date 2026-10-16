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
      tuple<sequence<1>, sequence<1>>,              // P0、P1 都映射到 X 维 1
      tuple<sequence<0>, sequence<1>>,              // ……的第 0 段、第 1 段
      sequence<1>,                                  // Y 映射到 X 维 1
      sequence<2>>{};                               // ……的第 2 段
  constexpr auto dist = make_static_tile_distribution(enc);

  // 3) tile window：本 block 负责 [blockIdx.x * kTile, +kTile)
  auto wa = make_tile_window(va, make_tuple(number<kTile>{}), {0}, dist);
  auto wb = make_tile_window(vb, make_tuple(number<kTile>{}), {0}, dist);
  auto wc = make_tile_window(vc, make_tuple(number<kTile>{}), {0}, dist);

  for (index_t tile = 0; tile * kTile < n; ++tile) {
    auto ta = load_tile(wa);
    auto tb = load_tile(wb);
    auto tc = make_static_distributed_tensor<float>(dist);
    sweep_tile(tc, [&](auto idx) { tc(idx) = ta[idx] + tb[idx]; });
    store_tile(wc, tc);
    move_tile_window(wa, {kTile});
    move_tile_window(wb, {kTile});
    move_tile_window(wc, {kTile});
  }
}

void launch_ck_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  const int grid = 1;
  ck_vector_add_kernel<<<grid, kBlock, 0, stream>>>(a, b, c, n);
}

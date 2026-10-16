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
  return transform_tensor_descriptor(make_col_major<M, N>(),
      make_tuple(make_xor_transform(make_tuple(number<M>{}, number<N>{}))),
      make_tuple(sequence<0, 1>{}),
      make_tuple(sequence<0, 1>{}));
}

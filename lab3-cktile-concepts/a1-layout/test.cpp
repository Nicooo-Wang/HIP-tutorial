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

// 设备端写出 descriptor 的上层维度数和各维长度：out[0] = 维度数，out[1..] = get_length(0..)。
// 偏移全对、长度写错的 descriptor 一旦拿去配 tile window / pad，越界屏蔽就会错，所以长度也是答案的一部分。
template <class MakeDesc, index_t NMax>
__global__ void dump_len(int* out) {
  constexpr auto d = MakeDesc{}();
  constexpr index_t n = d.get_num_of_top_dimension();
  out[0] = n;
  static_for<0, (n < NMax ? n : NMax), 1>{}([&](auto k) { out[1 + k] = d.get_length(k); });
}

// 期望长度 lens 给定；维度数不对时只报维度数
template <class MakeDesc>
static void check_lengths(hk::Result& R, const char* what, std::vector<int> lens) {
  constexpr index_t kMax = 4;
  hk::GuardedBuffer<int> out(1 + kMax);
  out.fill(-1);
  dump_len<MakeDesc, kMax><<<1, 1>>>(out.get());
  auto o = out.download();
  if (!out.guards_intact(R)) return;
  if (o[0] != int(lens.size())) { hk::fail(R, "%s：上层维度数期望 %d，得到 %d", what, int(lens.size()), o[0]); return; }
  for (size_t k = 0; k < lens.size(); ++k)
    if (o[1 + k] != lens[k]) {
      hk::fail(R, "%s：上层第 %d 维的长度（get_length）期望 %d，得到 %d（偏移对了不代表长度对）", what, int(k), lens[k], o[1 + k]);
      return;
    }
}

#define MAKER(name, fn) struct name { CK_TILE_HOST_DEVICE constexpr auto operator()() const { return fn; } }
MAKER(RowMajor4x8, (make_row_major<4, 8>()));
MAKER(RowMajor5x3, (make_row_major<5, 3>()));
MAKER(ColMajor4x8, (make_col_major<4, 8>()));
MAKER(ColMajor5x3, (make_col_major<5, 3>()));
MAKER(Transposed4x8, (make_transposed_view<4, 8>()));
MAKER(Transposed6x10, (make_transposed_view<6, 10>()));
MAKER(Transposed5x5, (make_transposed_view<5, 5>()));
MAKER(Xor8x8, (make_xor_swizzled<8, 8>()));
MAKER(Xor16x4, (make_xor_swizzled<16, 4>()));
MAKER(Flat4x8, (make_flattened<4, 8>()));
MAKER(Tiled, (make_tiled<8, 12, 4, 3>()));
MAKER(TiledSq, (make_tiled<12, 8, 4, 4>()));

TEST_CASE(row_major, 1) {
  check2d<RowMajor4x8, 4, 8>(R, "row_major<4,8>", [](int i, int j) { return i * 8 + j; });
  if (R.passed) check2d<RowMajor5x3, 5, 3>(R, "row_major<5,3>", [](int i, int j) { return i * 3 + j; });
  if (R.passed) check_lengths<RowMajor5x3>(R, "row_major<5,3>", {5, 3});
}
TEST_CASE(col_major, 1) {
  check2d<ColMajor4x8, 4, 8>(R, "col_major<4,8>", [](int i, int j) { return j * 4 + i; });
  if (R.passed) check2d<ColMajor5x3, 5, 3>(R, "col_major<5,3>", [](int i, int j) { return j * 5 + i; });
  if (R.passed) check_lengths<ColMajor4x8>(R, "col_major<4,8>", {4, 8});
  if (R.passed) check_lengths<ColMajor5x3>(R, "col_major<5,3>", {5, 3});
}
TEST_CASE(transposed_view, 2) {
  // 上层坐标 (j, i)：第 0 维长度 N，第 1 维长度 M
  check2d<Transposed4x8, 8, 4>(R, "transposed_view<4,8>（上层 8×4）", [](int j, int i) { return i * 8 + j; });
  if (R.passed) check2d<Transposed6x10, 10, 6>(R, "transposed_view<6,10>（上层 10×6）", [](int j, int i) { return i * 10 + j; });
  if (R.passed) check2d<Transposed5x5, 5, 5>(R, "transposed_view<5,5>", [](int j, int i) { return i * 5 + j; });
  if (R.passed) check_lengths<Transposed4x8>(R, "transposed_view<4,8>", {8, 4});
  if (R.passed) check_lengths<Transposed6x10>(R, "transposed_view<6,10>", {10, 6});
}
TEST_CASE(xor_swizzled, 2) {
  check2d<Xor8x8, 8, 8>(R, "xor_swizzled<8,8>", [](int i, int j) { return i * 8 + (j ^ (i % 8)); });
  if (R.passed) check2d<Xor16x4, 16, 4>(R, "xor_swizzled<16,4>", [](int i, int j) { return i * 4 + (j ^ (i % 4)); });
  if (R.passed) check_lengths<Xor16x4>(R, "xor_swizzled<16,4>", {16, 4});
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
  if (R.passed) check_lengths<Flat4x8>(R, "flattened<4,8>", {32});
}

template <class MakeDesc, index_t M, index_t N, index_t TM, index_t TN>
__global__ void dump_tiled(int* out) {
  constexpr auto d = MakeDesc{}();
  int p = 0;
  for (int I = 0; I < M / TM; ++I) for (int J = 0; J < N / TN; ++J) for (int ii = 0; ii < TM; ++ii) for (int jj = 0; jj < TN; ++jj)
    out[p++] = d.calculate_offset(make_tuple(I, J, ii, jj));
  out[p] = d.get_num_of_top_dimension();
}
template <class MakeDesc, int M, int N, int TM, int TN>
static void check_tiled(hk::Result& R, const char* what) {
  hk::GuardedBuffer<int> out(M * N + 1);
  out.fill(-1);
  dump_tiled<MakeDesc, M, N, TM, TN><<<1, 1>>>(out.get());
  std::vector<int> e;
  for (int I = 0; I < M / TM; ++I) for (int J = 0; J < N / TN; ++J) for (int ii = 0; ii < TM; ++ii) for (int jj = 0; jj < TN; ++jj)
    e.push_back((I * TM + ii) * N + J * TN + jj);
  e.push_back(4);
  if (hk::check_equal(R, out.download(), e, what)) out.guards_intact(R);
  if (R.passed) check_lengths<MakeDesc>(R, what, {M / TM, N / TN, TM, TN});
}
TEST_CASE(tiled, 3) {
  check_tiled<Tiled, 8, 12, 4, 3>(R, "tiled<8,12,4,3>，按 (I,J,ii,jj) 字典序排列（最后一项为上层维度数 4）");
  if (R.passed) check_tiled<TiledSq, 12, 8, 4, 4>(R, "tiled<12,8,4,4>，按 (I,J,ii,jj) 字典序排列（最后一项为上层维度数 4）");
}
HK_MAIN()

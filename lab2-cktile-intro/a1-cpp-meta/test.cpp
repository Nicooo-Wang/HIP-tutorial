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

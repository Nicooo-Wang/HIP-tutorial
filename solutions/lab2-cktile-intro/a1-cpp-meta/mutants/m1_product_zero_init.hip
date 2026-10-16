#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include <type_traits>
using ck_tile::index_t;

template <index_t... Is>
constexpr index_t product(ck_tile::sequence<Is...>) {
  return (index_t{0} * ... * Is);
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

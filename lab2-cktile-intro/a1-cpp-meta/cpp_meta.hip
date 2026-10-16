// Lab 2 · a1 cpp-meta：读懂 CK-Tile 所需的 C++（constexpr、变参模板、number、sequence、tuple、static_for、if constexpr、泛型 lambda）
// E1、E2、E4、E5 必须保持 constexpr：测试里会在编译期调用它们，丢掉 constexpr 整个测试会编译失败。
// （E3 是 __device__ 函数；E6 的 lambda 从 C++17 起本来就隐式 constexpr，保留骨架里的写法即可。）
// 判分：在 lab2-cktile-intro/ 目录下运行 make grade A=a1；调试：make run A=a1 CASE=用例名
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include <type_traits>
using ck_tile::index_t;

// ---------------- E1：sequence 中所有元素之积 ----------------
// 例：product(sequence<2,3,4>{}) == 24；空 sequence 返回 1。
// TODO(a1-E1): 用折叠表达式（fold expression）把 Is... 全部乘起来（讲义 §2.1）
//   同类例子（求和，你要改成乘积）：return (index_t{0} + ... + Is);   // 二元左折叠：初值 + 所有元素
//   想一想：乘法的“初值”应该是多少，空包时才对？
template <index_t... Is>
constexpr index_t product(ck_tile::sequence<Is...>) {
  // >>> 你的代码 <<<
  return 0;
}

// ---------------- E2：紧密排列的行主序 stride ----------------
// 例：<4,8,2> → {16,2,1}；<5> → {1}。返回 ck_tile::array<index_t, N>。
// TODO(a1-E2): 先把 Ls... 放进一个 constexpr 数组，再从最内层（最后一维）往外累乘，写出每一维的 stride（讲义 §2.2）
//   定义：行主序下，最后一维 stride 为 1；其余每一维的 stride = 它右边所有维度长度之积。
//   （作为对照：列主序是第一维 stride 为 1、每一维 stride = 它左边所有维度长度之积，方向正好相反。）
//   提示：constexpr 函数里可以用普通的局部变量和 for 循环；array 元素用 s(i) 写、s[i] 读。
template <index_t... Ls>
constexpr auto row_major_strides(ck_tile::sequence<Ls...>) {
  // >>> 你的代码 <<<
  return ck_tile::array<index_t, sizeof...(Ls)>{};
}

// ---------------- E3：设备端 static_for ----------------
// 要求：对 i = 0..N-1 写 out[i] = i * i（float）。
// TODO(a1-E3): 用 ck_tile::static_for<0, N, 1>{}(lambda) 展开循环（讲义 §2.3）
//   同类例子（写 out[i] = 2 * i）：
//     ck_tile::static_for<0, N, 1>{}([&](auto i) { out[i] = 2 * i; });
//   这里 i 的类型不是 int 而是 number<0>、number<1>……；它能隐式转成整数，也能通过 i.value 取得编译期常量。
//   请思考：把 i 当 int 用，和 i.value 当模板参数用，有什么区别？
template <index_t N>
__device__ void fill_squares(float* out) {
  // >>> 你的代码 <<<
}

// ---------------- E4：if constexpr ----------------
// 一次 16 字节向量访存能装多少个 T：float=4、half_t=8、bf16_t=8、int8_t=16、int32_t=4、double=2。
// TODO(a1-E4): 用 if constexpr 让 double 单独走一个分支、直接返回 2，其它类型按公式计算（讲义 §2.4）
//   （double 这个分支只是为了练习 if constexpr 的写法。想一想：去掉它，double 的结果会变吗？）
//   同类例子（数据类型是否为浮点，选不同返回值）：
//     if constexpr (std::is_floating_point_v<T>) return 1; else return 0;
//   if constexpr 的分支在编译期选择，没被选中的分支不会被实例化。
//   注意：sizeof(T) 的单位是字节还是比特？
template <class T>
constexpr index_t vec16() {
  // >>> 你的代码 <<<
  return 1;
}

// ---------------- E5：tuple 求和 ----------------
// 把 tuple 中所有元素（类型可以各不相同）转成 double 后求和：tuple_sum(make_tuple(1, 2.5f, 3.0)) == 6.5。
// TODO(a1-E5): 在 static_for 里用 t.at(i) 逐个取元素（讲义 §2.5）
//   同类例子（取第 0 个元素）：t.at(ck_tile::number<0>{})。
//   tuple 的每个元素类型不同，所以下标必须是编译期的 number<I>，不能是运行时的 int；static_for 给的循环变量正好就是。
template <class... Ts>
constexpr double tuple_sum(const ck_tile::tuple<Ts...>& t) {
  // >>> 你的代码 <<<
  return 0.0;
}

// ---------------- E6：泛型 lambda ----------------
// 两个 number<> 相加，结果仍是 number<>（值在类型里）：add_numbers(number<3>{}, number<4>{}) 的类型是 number<7>。
// TODO(a1-E6): 补全泛型 lambda（参数类型是 auto）的返回值（讲义 §2.6）
//   同类例子（单参数：返回 a 的两倍，仍放在类型里）：
//     [](auto a) { return ck_tile::number<decltype(a)::value * 2>{}; }
//   （这里的运算与你要写的不同，注意两个参数 a、b 的情况。）
//   想一想：a、b 作为函数参数是运行期的值，为什么 decltype(a)::value 仍能用在模板参数里？
constexpr auto add_numbers = [](auto a, auto b) {
  // >>> 你的代码 <<<
  return ck_tile::number<0>{};
};

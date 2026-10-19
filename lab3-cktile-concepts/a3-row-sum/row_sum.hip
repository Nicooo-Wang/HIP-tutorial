// Lab 3 · a3：用 CK-Tile 做行归约（Lab 3 收尾，自主设计）
// out[r] = Σ_c in[r*cols + c]，in 是 rows×cols 的行主序 fp32。
// 判分：在 lab3-cktile-concepts/ 目录下运行 make grade A=a3；调试：make run A=a3 CASE=用例名
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
using namespace ck_tile;

// TODO(a3-1): 自主设计 CK-Tile 行归约（讲义 §3.1–3.5）
//   需要决定：一个 block 处理几行、行/列分别映射到 warp/lane/Y 的哪一段、tile 宽度、怎样沿列滑动窗口。
//   必须遵守的约束（读 ck_tile/ops/reduce/block/block_reduce.hpp 第 9 行、第 12–15 行与第 257–260 行的注释）：
//     · block_tile_reduce 要求至少一个 Y 维不被归约
//     · block_tile_reduce_sync 只做 warp 内（跨 lane）归约 —— 所以一行不能跨 warp
//   边界：行数、列数都可能不是 tile 的整数倍（复习 a2 的 pad_tensor_view）。
//   性能：bandwidth_aligned 用例中 cols 是 4 的倍数；想想 make_naive_tensor_view 的倒数第二个参数什么时候可以写 4。
//   stream 原样传给 kernel 启动语句。
//   kernel 要你自己新增：写在下面这个标记处（文件作用域、launch_ck_row_sum 之前；模板 kernel 不能定义在函数体内），
//   个数、名字随你；再在 launch_ck_row_sum 里按 cols 分派、算好 grid 并启动它（讲义 §3.5）。
// >>> 你的 kernel（及辅助函数）<<<

void launch_ck_row_sum(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  // >>> 你的代码 <<<
}

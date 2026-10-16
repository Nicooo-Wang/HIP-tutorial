// Lab 1 · a2 矩阵转置：naive → LDS 分块
// in 是 rows×cols 行主序，out 是 cols×rows 行主序：out[c*rows + r] = in[r*cols + c]。
// Part A：launch_transpose_naive（每线程搬一个元素）；Part B：launch_transpose_lds（用 LDS 分块，让读写都合并）。
// 判分：在 lab1-hip-basics/ 目录下运行 make grade A=a2；调试：make run A=a2 CASE=用例名
#include <hip/hip_runtime.h>

// ---------------- Part A：naive ----------------

__global__ void transpose_naive_kernel(const float* in, float* out, int rows, int cols) {
  // TODO(a2-1): 由 2D 线程坐标得到 (r, c)，边界检查后写 out（讲义 §3.2）
  // >>> 你的代码 <<<
}

void launch_transpose_naive(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_naive_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

// ---------------- Part B：LDS 分块 ----------------

__global__ void transpose_lds_kernel(const float* in, float* out, int rows, int cols) {
  // TODO(a2-2): 设计你的分块转置（讲义 §3.4–3.6）
  //   第 0 步：决定 tile 大小与 block 形状（提示：一个 wavefront 是 64 线程，让它读一整行连续元素）
  //   第 1 步：声明 __shared__ 数组；想想为什么讲义建议多加 1 列
  //   第 2 步：从 in 合并地读入 tile（注意矩阵边缘的越界）
  //   第 3 步：同步——为什么必须在这里同步？
  //   第 4 步：从 LDS 按"转置后的方向"读出，合并地写入 out
  // >>> 你的代码 <<<
}

void launch_transpose_lds(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  // TODO(a2-3): 换成你的 LDS kernel 与对应的 grid/block
  launch_transpose_naive(in, out, rows, cols, stream);
  // >>> 你的代码 <<<
}

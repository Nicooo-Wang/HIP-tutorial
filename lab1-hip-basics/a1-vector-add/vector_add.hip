// Lab 1 · a1 vector add：向量加法与 grid-stride loop
// Part A：launch_vector_add 结束后 c[i] == a[i] + b[i]（grid 由你决定）。
// Part B：launch_saxpy 结束后 y[i] == alpha * x[i] + y[i]（grid 固定为 kSaxpyGrid 个 block，必须用 grid-stride loop）。
// 判分：在 lab1-hip-basics/ 目录下运行 make grade A=a1；调试：make run A=a1 CASE=用例名
#include <hip/hip_runtime.h>

// ---------------- Part A：c = a + b ----------------

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  // TODO(a1-1): 计算本线程负责的全局下标 i，做边界检查（i < n），然后写 c[i] = a[i] + b[i]（讲义 §2.3）
  // >>> 你的代码 <<<
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = b[idx] + a[idx];
  }
}

void launch_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  // TODO(a1-2): 自己选择 block 大小，并向上取整计算 grid，使线程总数 >= n（讲义 §2.4）
  // >>> 你的代码 <<<
  const int block = 64;                     // ← 可以改
  const int grid = (n + block - 1) / block; // ← 换成正确的表达式
  vector_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
}

// ---------------- Part B：y = alpha * x + y ----------------

constexpr int kSaxpyGrid = 160;  // 不要修改：saxpy 的 grid 固定为这么多个 block

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
  // TODO(a1-3): 全网格只有 kSaxpyGrid * 256 个线程，n 可能远大于它。
  //             用 grid-stride loop 让每个线程处理多个元素，覆盖所有 n（讲义 §2.5）
  // >>> 你的代码 <<<

  auto tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t ids = tid; ids < n; ids += stride) {
    y[ids] = alpha * x[ids] + y[ids];
  }
}

void launch_saxpy(float alpha, const float* x, float* y, int n, hipStream_t stream) {
  saxpy_kernel<<<kSaxpyGrid, 256, 0, stream>>>(alpha, x, y, n);  // 不要修改 grid
}

// Lab 1 · a4 GEMM：naive → LDS 分块
// C[M×N] = A[M×K] · B[K×N]，全部行主序 fp32。
// 判分：在 lab1-hip-basics/ 目录下运行 make grade A=a4；调试：make run A=a4 CASE=用例名
#include <hip/hip_runtime.h>

// ---------------- Part A：naive（每个线程算 C 的一个元素） ----------------
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  // TODO(a4-1): 每个线程计算 C 的一个元素（讲义 §5.1）。想想：让相邻线程对应相邻的 j 还是相邻的 i？
  // >>> 你的代码 <<<
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((N + 63) / 64, (M + 3) / 4);
  gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// ---------------- Part B：LDS 分块（自主设计） ----------------
// TODO(a4-2): 自主设计 LDS 分块 GEMM（讲义 §5.2–5.4）
//   在这里（launch_gemm_tiled 之前）新增你自己的 __global__ kernel（个数、名字随你），再在下面的函数里启动它。
//   你需要决定：tile 尺寸、block 形状、每个线程算几个输出、LDS 怎么摆放、K 方向怎么循环、边界怎么处理。
//   可复用 a2 学到的 padding、a3 学到的同步。目标：tiled_gflops 用例的性能门槛（见 test.cpp 中 kTiledGflops）。
//   设计完成后，在 ../answers.md 的"设计说明"处用 3–5 句话写下你的选择和理由。
void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  // >>> 你的代码 <<<
}

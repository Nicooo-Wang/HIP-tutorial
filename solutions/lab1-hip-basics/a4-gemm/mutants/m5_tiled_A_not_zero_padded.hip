#include <hip/hip_runtime.h>

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列：相邻线程访问 B、C 的相邻元素 → 合并访存
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[size_t(i) * K + k] * B[size_t(k) * N + j];
  C[size_t(i) * N + j] = acc;
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((N + 63) / 64, (M + 3) / 4);
  gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// 分块：block 计算 C 的 TILE×TILE 子块；256 线程 = 32 列 × 8 行，每线程计算同一列上 TILE/8 = 4 个元素。
constexpr int TILE = 32;
constexpr int TY = 8;
constexpr int RPT = TILE / TY;  // rows per thread

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row0 = blockIdx.y * TILE, col = blockIdx.x * TILE + tx;
  float acc[RPT] = {};
  for (int k0 = 0; k0 < K; k0 += TILE) {
    for (int r = 0; r < RPT; ++r) {
      int lr = ty + r * TY;
      int ar = row0 + lr, ac = k0 + tx;
      As[lr][tx] = A[size_t(ar) * K + ac];  // BUG：越界时不补 0（越界读，乘以补 0 的 Bs 也抵消不了 NaN）
      int br = k0 + lr;
      Bs[lr][tx] = (br < K && col < N) ? B[size_t(br) * N + col] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < TILE; ++kk) {
      float b = Bs[kk][tx];
      for (int r = 0; r < RPT; ++r) acc[r] += As[ty + r * TY][kk] * b;
    }
    __syncthreads();  // 下一轮覆盖 As/Bs 前，所有线程必须用完当前 tile
  }
  for (int r = 0; r < RPT; ++r) {
    int i = row0 + ty + r * TY;
    if (i < M && col < N) C[size_t(i) * N + col] = acc[r];
  }
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t stream) {
  dim3 block(TILE, TY);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

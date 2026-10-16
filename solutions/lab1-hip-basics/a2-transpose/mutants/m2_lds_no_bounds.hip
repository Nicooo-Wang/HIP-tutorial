#include <hip/hip_runtime.h>

__global__ void transpose_naive_kernel(const float* in, float* out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[size_t(c) * rows + r] = in[size_t(r) * cols + c];
}

void launch_transpose_naive(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 block(64, 4);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_naive_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

constexpr int TILE = 64;      // 每个 block 处理 TILE×TILE 的子块
constexpr int ROWS_PER = 4;   // block = TILE × ROWS_PER 线程，每线程处理 TILE/ROWS_PER 行

__global__ void transpose_lds_kernel(const float* in, float* out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];  // +1 padding 消除 bank conflict
  int c0 = blockIdx.x * TILE, r0 = blockIdx.y * TILE;
  // 1) 合并地读入：同一 wavefront 的线程读同一行的连续元素
  for (int dy = threadIdx.y; dy < TILE; dy += ROWS_PER) {
    int r = r0 + dy, c = c0 + threadIdx.x;
    if (true) tile[dy][threadIdx.x] = in[size_t(r) * cols + c];
  }
  __syncthreads();
  // 2) 合并地写出：输出的一行对应输入的一列，从 LDS 按列读取
  for (int dy = threadIdx.y; dy < TILE; dy += ROWS_PER) {
    int oc = r0 + threadIdx.x;  // 输出列 = 输入行
    int orow = c0 + dy;         // 输出行 = 输入列
    if (true) out[size_t(orow) * rows + oc] = tile[threadIdx.x][dy];
  }
}

void launch_transpose_lds(const float* in, float* out, int rows, int cols, hipStream_t stream) {
  dim3 block(TILE, ROWS_PER);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transpose_lds_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

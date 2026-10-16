#include <hip/hip_runtime.h>

__global__ void fill_ids_kernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = i;  // mutant: 去掉边界检查
}

void launch_fill_ids(int* out, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  fill_ids_kernel<<<grid, block, 0, stream>>>(out, n);
}

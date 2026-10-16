#include <hip/hip_runtime.h>

__global__ void fill_ids_kernel(int* out, int n) {
  int i = threadIdx.x;  // mutant: 只用块内编号，所有 block 写同一段
  if (i < n) out[i] = i;
}

void launch_fill_ids(int* out, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  fill_ids_kernel<<<grid, block, 0, stream>>>(out, n);
}

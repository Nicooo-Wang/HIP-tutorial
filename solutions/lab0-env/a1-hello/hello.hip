#include <hip/hip_runtime.h>

__global__ void fill_ids_kernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

void launch_fill_ids(int* out, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  fill_ids_kernel<<<grid, block, 0, stream>>>(out, n);
}

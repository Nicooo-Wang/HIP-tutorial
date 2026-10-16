#include <hip/hip_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  float fi = float(blockIdx.x * blockDim.x + threadIdx.x); int i = int(fi);
  if (i < n) c[i] = a[i] + b[i];
}

void launch_vector_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  vector_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
}

constexpr int kSaxpyGrid = 160;

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = alpha * x[i] + y[i];
}

void launch_saxpy(float alpha, const float* x, float* y, int n, hipStream_t stream) {
  saxpy_kernel<<<kSaxpyGrid, 256, 0, stream>>>(alpha, x, y, n);
}

#include <hip/hip_runtime.h>

constexpr int kWorkspaceFloats = 4096;
constexpr int kBlock = 256;
constexpr int kMaxBlocks = 1024;  // ≤ kWorkspaceFloats

// ---------------- LDS 树形归约 ----------------
__device__ float block_reduce_lds(float v) {
  __shared__ float s[kBlock];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  return s[0];  // 所有线程都能读到
}

// ---------------- wavefront shuffle 归约 ----------------
__device__ float wave_reduce(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
  return v;  // lane 0 持有整个 wavefront 的和
}

__device__ float block_reduce_shfl(float v) {
  __shared__ float partial[kBlock / 64];
  int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  v = wave_reduce(v);
  if (lane == 0) partial[wave] = v;
  __syncthreads();
  v = (threadIdx.x < kBlock / warpSize) ? partial[threadIdx.x] : 0.0f;
  if (wave == 0) v = wave_reduce(v);
  return v;  // thread 0 持有结果
}

template <bool kShfl>
__global__ void partial_sum_kernel(const float* in, int n, float* dst) {
  float acc = 0.0f;
  for (int i = blockIdx.x * kBlock + threadIdx.x; i < n; i += gridDim.x * kBlock) acc += in[i];
  float s = kShfl ? block_reduce_shfl(acc) : block_reduce_lds(acc);
  if (threadIdx.x == 0) atomicAdd(dst, s);  // 直接累加到 out，且从未清零
}

template <bool kShfl>
static void reduce_two_pass(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  int blocks = (n + kBlock - 1) / kBlock;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  partial_sum_kernel<kShfl><<<blocks, kBlock, 0, stream>>>(in, n, out);
}

void reduce_sum_lds(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<false>(in, n, out, ws, stream);
}
void reduce_sum_shfl(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<true>(in, n, out, ws, stream);
}

// Lab 1 · a3 归约：LDS 树形 → wavefront shuffle
// *out = sum(in[0..n))。ws 是测试提供的 kWorkspaceFloats 个 float 的设备工作区，内容未初始化；out 的初值也不可信（可能是 NaN）。
// 两遍结构（已给出）：
//     in --(多 block，每个 block 出一个部分和)--> ws[0..blocks) --(1 个 block 汇总)--> out
// block 之间没有 __syncthreads，最直接的做法是分成两次 kernel 启动（kernel 边界就是全局屏障，讲义 §4.1）。
// 判分：在 lab1-hip-basics/ 目录下运行 make grade A=a3；调试：make run A=a3 CASE=用例名
#include <hip/hip_runtime.h>

constexpr int kWorkspaceFloats = 4096;
constexpr int kBlock = 256;
constexpr int kMaxBlocks = 1024;  // ≤ kWorkspaceFloats

// ---------------- Part A：LDS 树形归约 ----------------
__device__ float block_reduce_lds(float v) {
  // TODO(a3-2): block_reduce_lds：用 __shared__ 数组做树形归约，每一层之后需要什么？（讲义 §4.2）
  //   约定：返回 kBlock 个线程的 v 之和（至少 thread 0 必须拿到正确结果）
  // >>> 你的代码 <<<
  return 0.0f;
}

// ---------------- Part B：wavefront shuffle 归约 ----------------
__device__ float wave_reduce(float v) {
  // TODO(a3-3): wave_reduce：用 __shfl_down 在一个 wavefront 内归约。
  //   注意 AMD 的 wavefront 是 64 线程，用 warpSize 而不是写死 32（讲义 §4.3）
  //   约定：lane 0 返回整个 wavefront 的和
  // >>> 你的代码 <<<
  return v;
}

__device__ float block_reduce_shfl(float v) {
  // TODO(a3-4): block_reduce_shfl：先 wave 内归约，再把每个 wave 的结果放进 LDS，最后由第 0 个 wave 再归约一次（讲义 §4.4）
  //   约定：thread 0 返回整个 block 的和
  // >>> 你的代码 <<<
  return v;
}

template <bool kShfl>
__global__ void partial_sum_kernel(const float* in, int n, float* dst) {
  float acc = 0.0f;
  // TODO(a3-1): 每个线程用 grid-stride loop 累加自己负责的元素到 acc（复习 a1 Part B）
  // >>> 你的代码 <<<
  float s = kShfl ? block_reduce_shfl(acc) : block_reduce_lds(acc);
  if (threadIdx.x == 0) dst[blockIdx.x] = s;
}

template <bool kShfl>
static void reduce_two_pass(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  int blocks = (n + kBlock - 1) / kBlock;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  partial_sum_kernel<kShfl><<<blocks, kBlock, 0, stream>>>(in, n, ws);   // 第 1 遍：每 block 一个部分和
  partial_sum_kernel<kShfl><<<1, kBlock, 0, stream>>>(ws, blocks, out);  // 第 2 遍：汇总部分和
}

void reduce_sum_lds(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<false>(in, n, out, ws, stream);
}
void reduce_sum_shfl(const float* in, int n, float* out, float* ws, hipStream_t stream) {
  reduce_two_pass<true>(in, n, out, ws, stream);
}

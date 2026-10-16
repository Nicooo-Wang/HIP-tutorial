// Lab 0 · a1 hello：你的第一个 kernel
// 任务：launch_fill_ids(out, n, stream) 结束后，out[i] == i 对所有 0 <= i < n 成立。
// 判分：在 lab0-env/ 目录下运行 make grade A=a1；调试：make run A=a1 CASE=用例名
#include <hip/hip_runtime.h>

// 每个 GPU 线程负责 out 中的（至多）一个元素。
__global__ void fill_ids_kernel(int* out, int n) {
  // TODO(a1-1): 计算本线程负责的全局下标 i，并在 i < n 时写 out[i] = i（讲义 §3.2）
  //   提示：要用到 blockIdx.x、blockDim.x、threadIdx.x 三个内置变量（讲义 §3.2 的“楼与房间”类比）
  // >>> 你的代码 <<<
}

// 主机端函数：由 test.cpp 调用。out 已经是 GPU 显存上的指针（长度 n）。
void launch_fill_ids(int* out, int n, hipStream_t stream) {
  const int block = 256;  // 每个 block 256 个线程（= 4 个 wavefront）
  // TODO(a1-2): grid 要足够大以覆盖全部 n 个元素（向上取整），见讲义 §3.3
  // >>> 你的代码 <<<
  const int grid = 1;  // ← 把 1 换成正确的表达式
  fill_ids_kernel<<<grid, block, 0, stream>>>(out, n);
}

# Lab 0 书面问题

> 书面问题不计分，但都是后面几课要用到的直觉。题目原文见各 Assignment 的 README。

## a1 hello（a1-hello/README.md）

### 1. n = 1000、blockDim = 256 时启动了多少个线程？多出来的线程在做什么？

启动了4个线程，多出来的线程在空转

### 2. 去掉 `if (i < n)` 后，n = 1000 时会发生什么？为什么“结果看起来对”不代表程序正确？

如果去掉的话，实际上多出来的 1024 那 24 个线程实际上还是会写入。会写入到这个 Memory 之后，其实会导致错误写入脏数据

## a2 device-query（a2-device-query/README.md）

### 1. 本机 GPU 有多少个 CU？

80

### 2. 每个 CU 理论上最多同时驻留多少个 wavefront？整张 GPU 最多同时驻留多少个 wavefront、多少个线程？

每个 CU 最多能驻留 32 个微 front，整个 GPU 最多同时驻留 80×32 个微 front，线程数量是 80×32×64

### 3. 1024 个线程的 block 由几个 wavefront 组成？blockDim = 100 时呢（最后一个 wavefront 有多少个 lane 空闲）？

1024 个线程的 Block 由1024 / 64 wave Front 组成。当 Block Dimension 等于 100 的时候，应该是由两个 wave Front 组成。最后一个 Wave Front 有 28 个 lane 空闲

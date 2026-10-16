# Lab 0 书面问题

> 书面问题不计分，但都是后面几课要用到的直觉。题目原文见各 Assignment 的 README。

## a1 hello（a1-hello/README.md）

### 1. n = 1000、blockDim = 256 时启动了多少个线程？多出来的线程在做什么？

（在此作答）

### 2. 去掉 `if (i < n)` 后，n = 1000 时会发生什么？为什么“结果看起来对”不代表程序正确？

（在此作答）

## a2 device-query（a2-device-query/README.md）

### 1. 本机 GPU 有多少个 CU？

（在此作答）

### 2. 每个 CU 理论上最多同时驻留多少个 wavefront？整张 GPU 最多同时驻留多少个 wavefront、多少个线程？

（在此作答）

### 3. 1024 个线程的 block 由几个 wavefront 组成？blockDim = 100 时呢（最后一个 wavefront 有多少个 lane 空闲）？

（在此作答）

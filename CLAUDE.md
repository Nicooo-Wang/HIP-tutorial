# HIP / CK-Tile 学习仓库 —— 项目约定

## 定位
- 学习者：了解 GPU 概念、未系统写过 kernel；会 C++ 但模板元编程不熟；目标是在 MI308X 上做 mega kernel 研究。
- 一律用中文交流。用户常用语音输入："Hive"=HIP，"Cute/Catalyst/CkTile"≈CK-Tile，不确定时确认。
- 第一优先级：学习连续性——每个 Assignment 只用此前 Tutorial 讲过的概念。

## 结构
- 每个 Lab 一个文件夹 `labN-<name>/`，内含 `tutorial.html`、`Makefile`、`answers.md`、`aK-<name>/{<name>.hip,test.cpp,README.md}`。
- 参考答案与 mutants：`solutions/labN-<name>/aK-<name>/{<name>.hip,mutants/*.hip}`。mutants 绝不放进学生目录。
- 设计文档：`docs/superpowers/specs/`；计划：`docs/superpowers/plans/`；审查报告：`docs/reviews/`。

## 挖空规范
- 难度 1 填函数体 / 难度 2 补全设计 / 难度 3 自主设计，随课程推进过渡。
- 格式：`// TODO(aK-n): <中文说明>（讲义 §x.y）` + `// >>> 你的代码 <<<`。
- README：任务、分级 Hints（Hint 1 方向、Hint 2 关键步骤，不给代码）、书面问题。
- 骨架必须能编译，且原样提交时至少一个用例 FAIL。

## 判分规范
- test.cpp 通过 `#include STUDENT_SRC` 引入学生文件，只依赖 `common/harness.hpp`。
- 正确性：多尺寸（含 1、非整除、较大）、多种子、容差按类型、GuardedBuffer 检测越界、输入不被修改。
- 性能：计入分数；阈值校准（参考答案与慢 mutant 各留 ≥1.5× 余量）；内存类以同进程 D2D 拷贝带宽为基准。
- `make verify` 全绿（solutions 全过、每个 mutant 至少挂一项、骨架不满分）才算 Assignment 完成。
- mutant 必须确定性失败，不依赖竞态。

## 环境
- ROCm 7.2 `/opt/rocm`，8× MI308X gfx942（80 CU、wave64、LDS 64 KiB）。CK-Tile：`/opt/rocm/include/ck_tile`。无 numpy/torch。
- 编译：`/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942`。
- 机器多人共享：grade 自动选空闲 GPU，用例 60 s 超时，单次显存 ≤ 2 GB。

## CK-Tile 已知坑（实测）
- `tile_distribution_encoding` 只有 1 个 P 维时，P 只对应 lane id（0–63），整个 block 只有一个 wavefront 生效；用在多 wavefront block 上能编译、能运行，只是静默出错（编译期只拦 NDimP ∉ {1,2}）。block 级分布要用 2 个 P 维（warp id, lane id）。
- `make_tile_window` 的原点用 `{static_cast<index_t>(blockIdx.x) * Tile}`，否则 unsigned→int 窄化报错。
- `make_naive_tensor_view` 省略最后两个 guarantee 参数（向量长度、最后一维 stride）时 CK-Tile 不会推导，`ScalarPerVector` = 1，逐元素访问；1D float 例子汇编相同只是后端合并了标量 buffer 访存，不可依赖。
- buffer 指令越界检查粒度是 4 字节（dword）：fp16/bf16/int8 等小类型在末尾部分越界的 dword 里，合法元素也会读成 0、写被丢弃（fp16 n=505 实测）。
- naive 视图的越界屏蔽只靠硬件 range = `element_space_size × sizeof(T)` 字节（带行间距的视图是 (rows−1)·ld + cols 个元素，不是 rows·cols），软件 is_valid 标志恒为 true。2D tile 只有内层维（列）越界会“绕”出本行：带行间距（ld > cols）的视图里，小幅越界落在行间空位，越界超过 ld − cols 才读写到下一行的数据（ld = cols 时一越界就是下一行）；超出最后一行末尾的部分（包括最后一行的列越界和整行越界）超出 range，被硬件屏蔽。边缘要用 `pad_tensor_view`（33×31 转置不 pad 实测出错）。
- `block_tile_reduce` 只支持 2D tile 归约第 1 维（`sequence<1>{}`）；写 `sequence<0>{}` 能编译、能运行，结果静默错误。`block_tile_reduce_sync` 只合并 lane 认领的 R 段，不跨 wavefront。
- 视图承诺向量长度 V 时，pad 只检查向量第一个元素：`cols % V ≠ 0` 时行尾向量会读进下一行（gfx942 上非 16 字节对齐的 dwordx4 照样读对，可见后果是跨行，不是对齐）。

## Tutorial 风格
- 共用 `common/tutorial.css` / `common/tutorial.js`；左侧目录、代码高亮、SVG 图示、交互小部件、浅色/深色模式；每节末尾标注对应 Assignment。
- 外部脚本仅 cdnjs 且固定版本。

### 行文规范
读者是“定位”里的学习者。新写或改写讲义正文时遵守：
- **每段先给结论**：一段只讲一件事，第一句说结论或要做的事，再讲原因和细节。先讲“是什么、怎么用”，再讲“为什么”；例外放最后或放进折叠框。
- **句子要短**：一句只表达一个意思；要用两个以上“但、因为、所以、而”才说得完的，拆开。括号插入语最多一层，只放补充说明（单位、出处、英文原名、一句话定义），不放结论或条件；不用破折号套括号。指代可能不清时重复名词，少用“它、这、该、其”。
- **术语**：用词以 `docs/glossary.md` 为准。HIP/AMD 通用英文术语保留英文（block、wavefront、lane、LDS、CU、kernel 等），不自造译名；有通用中文说法的用中文（归约、合并访存、屏障等）。术语第一次出现时用一句话解释；代码里的名字第一次出现时说明它是什么，如“工作区 `ws`”；引用骨架常量时写出名字和值，如 `kMaxBlocks = 1024`，便于对照代码。
- **只用学过的概念**：非提不可时，一句话交代它是什么，并注明“§x.y 详讲”或“Lab N 详讲”。
- **用例子说话**：抽象结论配具体数字或例子，优先用 MI308X 和本 Lab 作业里的数。数字推导写出中间步骤，不让读者心算。
- **严谨但不堆砌**：正文写学习者此时需要的结论和成立条件，必须正确。次要的例外、实测细节、背景原因放进折叠框 `<details class="deep"><summary>一句话说明里面讲什么</summary>…</details>`；不读折叠框也能看懂正文，正文也不会因此出错。summary 用问句或结论句；正文指向折叠框时说清里面讲什么。实测数字只保留支撑结论所需的几个，并注明条件（哪张卡、是否空闲、规模、重复次数），条件可放进折叠框。
- **保持不变**：§ 编号和标题 id 不变（README 和骨架里的“讲义 §x.y”引用依赖它们）；图示、小部件及其元素 id 不变。

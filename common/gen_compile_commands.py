#!/usr/bin/env python3
"""生成仓库根目录的 compile_commands.json，供 clangd（VSCode 代码补全/跳转）使用。

编译参数与 lab.mk 的 hipcc 一致（hipcc 实际调用的就是 ROCm 的 clang++ --offload-arch=gfx942）：
  - labN-*/aK-*/*.hip 与 solutions/ 下的 .hip（含 mutants）：test.cpp 会先 include harness.hpp
    再 include 学生文件，所以单独解析 .hip 时用 -include harness.hpp 模拟（如 HIP_CHECK）；
  - labN-*/aK-*/test.cpp：按 HIP 编译，STUDENT_SRC 指向同目录的学生 .hip；
  - common/harness_selftest.cpp。
新增 Lab / Assignment 后重新运行：make compdb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "common"
CLANG = "/opt/rocm/llvm/bin/clang++"
FLAGS = ["-x", "hip", "--offload-arch=gfx942", "--rocm-path=/opt/rocm",
         "-std=c++20", "-O2", f"-I{COMMON}"]
HIP_EXTRA = ["-include", str(COMMON / "harness.hpp")]


def entry(src, extra=()):
    return {"directory": str(src.parent), "file": str(src),
            "arguments": [CLANG, *FLAGS, *extra, "-c", str(src)]}


def main():
    entries = []
    for test in sorted(ROOT.glob("lab*/a*-*/test.cpp")):
        hips = sorted(test.parent.glob("*.hip"))
        for h in hips:
            entries.append(entry(h, HIP_EXTRA))
        if len(hips) == 1:
            entries.append(entry(test, [f'-DSTUDENT_SRC="{hips[0]}"']))
    for h in sorted((ROOT / "solutions").rglob("*.hip")):
        entries.append(entry(h, HIP_EXTRA))
    entries.append(entry(COMMON / "harness_selftest.cpp"))
    out = ROOT / "compile_commands.json"
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"已写入 {out.relative_to(ROOT)}（{len(entries)} 条）")


if __name__ == "__main__":
    main()

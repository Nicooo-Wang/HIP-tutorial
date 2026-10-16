#!/usr/bin/env python3
"""校验判分器：参考答案必须满分；每个 mutant 至少挂一项；学生骨架必须能编译且不满分。"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade  # noqa: E402

MIN_MUTANTS = 2


def verify_lab(lab_dir, sol_root):
    lab_dir = Path(lab_dir).resolve()
    problems = []
    for a in grade.discover(lab_dir):
        sdir = Path(sol_root) / lab_dir.name / a.name
        src_name = grade.student_src(a).name
        print(f"== {a.name}")
        sol_src = sdir / src_name
        if not sol_src.exists():
            problems.append(f"{a.name}: 缺少参考答案 {sol_src}")
        else:
            sol = grade.grade_assignment(a, src=sol_src, quiet=True)
            if not sol["built"]:
                problems.append(f"{a.name}: 参考答案编译失败\n{sol['build_log']}")
            elif sol["score"] != sol["max"]:
                bad = [f"{c['name']}: {c['msg']}" for c in sol["cases"] if not c["passed"]]
                problems.append(f"{a.name}: 参考答案未满分 {sol['score']}/{sol['max']}\n  " + "\n  ".join(bad))
            else:
                perf = [f"{c['name']}={c['metric']:.1f}{c['unit']}" for c in sol["cases"] if c["perf"]]
                print(f"   solution    满分 {sol['score']}/{sol['max']} {' '.join(perf)}")
        mutants = sorted((sdir / "mutants").glob("*.hip"))
        if len(mutants) < MIN_MUTANTS:
            problems.append(f"{a.name}: mutant 数量 {len(mutants)} < {MIN_MUTANTS}")
        for m in mutants:
            r = grade.grade_assignment(a, src=m, quiet=True)
            if not r["built"]:
                problems.append(f"{a.name}: mutant {m.name} 编译失败（mutant 必须能编译）")
                continue
            failed = [c["name"] for c in r["cases"] if not c["passed"]]
            if not failed:
                problems.append(f"{a.name}: mutant {m.name} 未被任何用例抓住")
            else:
                print(f"   mutant      {m.name:<32} 被抓住: {', '.join(failed)}")
        sk = grade.grade_assignment(a, quiet=True)
        if not sk["built"]:
            problems.append(f"{a.name}: 学生骨架无法编译\n{sk['build_log']}")
        elif sk["score"] >= sk["max"]:
            problems.append(f"{a.name}: 学生骨架原样即满分")
        else:
            print(f"   skeleton    {sk['score']}/{sk['max']}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lab_dir")
    ap.add_argument("--solutions-root", default=str(grade.REPO / "solutions"))
    args = ap.parse_args()
    grade.select_gpu()
    problems = verify_lab(args.lab_dir, args.solutions_root)
    if problems:
        print("\nVERIFY FAILED:")
        for p in problems:
            print(" - " + p)
        sys.exit(1)
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()

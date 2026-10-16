#!/usr/bin/env python3
"""判分驱动：编译每个 Assignment，逐个用例在独立进程中运行，汇总分数。仅用标准库。"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
REPO = COMMON.parent
HIPCC = "/opt/rocm/bin/hipcc"
CXXFLAGS = ["-std=c++20", "-O2", "--offload-arch=gfx942", f"-I{COMMON}"]
PERF_TRIES = 3
BUILD_TIMEOUT = 600
_RUN = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def timeout_s():
    return float(os.environ.get("HK_TIMEOUT", "60"))


def discover(lab_dir):
    lab_dir = Path(lab_dir)
    return sorted(p for p in lab_dir.iterdir()
                  if p.is_dir() and re.match(r"a\d+-", p.name) and (p / "test.cpp").exists())


def student_src(assign_dir):
    hips = sorted(Path(assign_dir).glob("*.hip"))
    if len(hips) != 1:
        raise RuntimeError(f"{assign_dir} 中应恰好有一个 .hip 文件，实际 {len(hips)} 个")
    return hips[0]


def build(assign_dir, src, out_bin):
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [HIPCC, *CXXFLAGS, f'-DSTUDENT_SRC="{Path(src).resolve()}"',
           str(Path(assign_dir).resolve() / "test.cpp"), "-o", str(out_bin)]
    try:
        p = subprocess.run(cmd, timeout=BUILD_TIMEOUT, **_RUN)
    except subprocess.TimeoutExpired:
        return False, f"编译超时（>{BUILD_TIMEOUT} s）"
    except OSError as e:
        return False, f"无法运行 hipcc: {e}"
    return p.returncode == 0, p.stdout + p.stderr


def select_gpu():
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showuse", "--showmemuse", "--json"],
                             timeout=30, **_RUN).stdout
        data = json.loads(out[out.index("{"):])
    except Exception:
        return
    best = None
    for card, info in data.items():
        m = re.match(r"card(\d+)$", card)
        if not m:
            continue
        try:
            load = float(info.get("GPU use (%)", 100)) + float(info.get("GPU Memory Allocated (VRAM%)", 100))
        except (ValueError, TypeError, AttributeError):
            continue
        if best is None or load < best[0]:
            best = (load, m.group(1))
    if best:
        os.environ["HIP_VISIBLE_DEVICES"] = best[1]


def gpu_load_note():
    """性能用例失败时附上当前所用 GPU 的占用，帮助区分"代码慢"与"别人在用卡"。"""
    dev = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi", "-d", dev, "--showuse", "--showmemuse", "--json"],
                             timeout=30, **_RUN).stdout
        info = next(iter(json.loads(out[out.index("{"):]).values()))
        return f"（当前 GPU {dev} 占用：计算 {info.get('GPU use (%)')}%，显存 {info.get('GPU Memory Allocated (VRAM%)')}%）"
    except Exception:
        return ""


def _crash(p_rc, text):
    tail = "\n".join(text.strip().splitlines()[-15:])
    return {"passed": False, "msg": f"进程崩溃（退出码 {p_rc}）\n{tail}", "metric": 0, "unit": ""}


def run_case(binary, name):
    try:
        p = subprocess.run([str(binary), "--case", name], timeout=timeout_s(), **_RUN)
    except subprocess.TimeoutExpired:
        return {"passed": False, "msg": f"超时（>{timeout_s():.0f} s）：可能死循环、死锁或越界导致挂起", "metric": 0, "unit": ""}
    if p.returncode == 0:
        # 取最后一条 @@RESULT（学生代码即使自行打印也会被 harness 的最终结果覆盖），且用例名必须匹配
        for line in reversed(p.stdout.splitlines()):
            if line.startswith("@@RESULT "):
                try:
                    r = json.loads(line[len("@@RESULT "):])
                    if r.get("name") != name:
                        continue
                    return {"passed": r["passed"] is True, "msg": str(r.get("msg", "")),
                            "metric": float(r.get("metric", 0)), "unit": str(r.get("unit", ""))}
                except (ValueError, KeyError, TypeError):
                    break
    return _crash(p.returncode, p.stdout + p.stderr)


def grade_assignment(assign_dir, src=None, quiet=False):
    assign_dir = Path(assign_dir)
    src = Path(src) if src else student_src(assign_dir)
    binary = assign_dir.parent / "build" / assign_dir.name / f"test-{os.getpid()}-{abs(hash(str(src))) % 10**8}"
    res = {"name": assign_dir.name, "built": False, "build_log": "", "cases": [], "score": 0, "max": 0}
    ok, log = build(assign_dir, src, binary)
    res["built"], res["build_log"] = ok, log
    if not ok:
        res["max"] = sum(c["points"] for c in list_cases_from_source(assign_dir))
        if not quiet:
            print(f"{assign_dir.name}: 编译失败（0 分）\n{log}")
        return res
    try:
        try:
            lp = subprocess.run([str(binary), "--list"], timeout=timeout_s(), **_RUN)
            cases = json.loads(lp.stdout)
        except (subprocess.TimeoutExpired, ValueError):
            res["built"] = False
            res["build_log"] = "无法列出用例（--list 超时或输出无效）。请检查是否在全局作用域/静态初始化中运行了代码。"
            res["max"] = sum(c["points"] for c in list_cases_from_source(assign_dir))
            if not quiet:
                print(f"{assign_dir.name}: {res['build_log']}")
            return res
        if not quiet:
            print(f"{assign_dir.name}")
        for c in cases:
            r = run_case(binary, c["name"])
            if c["perf"] and not r["passed"]:
                for _ in range(PERF_TRIES - 1):
                    r2 = run_case(binary, c["name"])
                    if r2["passed"] or r2["metric"] > r["metric"]:
                        r = r2
                    if r["passed"]:
                        break
                if not r["passed"]:
                    r["msg"] += f"\n已重试 {PERF_TRIES} 次{gpu_load_note()}"
            case = {**c, **r}
            res["cases"].append(case)
            res["max"] += c["points"]
            res["score"] += c["points"] if r["passed"] else 0
            if not quiet:
                perf = f"  [{r['metric']:.1f} {r['unit']}]" if c["perf"] else ""
                status = "OK" if r["passed"] else "FAIL"
                print(f"  {c['name']:.<44} {status} ({c['points'] if r['passed'] else 0}/{c['points']}){perf}")
                if not r["passed"] and r["msg"]:
                    print("    " + r["msg"].replace("\n", "\n    "))
    finally:
        try:
            binary.unlink()
        except OSError:
            pass
    return res


def list_cases_from_source(assign_dir):
    """编译失败时从 test.cpp 源码估算满分，便于显示 0/满分。"""
    text = (Path(assign_dir) / "test.cpp").read_text(encoding="utf-8")
    return [{"name": m.group(2), "points": int(m.group(3))}
            for m in re.finditer(r"\b(TEST_CASE|PERF_CASE)\((\w+),\s*(\d+)\)", text)]


def main():
    ap = argparse.ArgumentParser(description="判分")
    ap.add_argument("lab_dir")
    ap.add_argument("--only", help="只判某个 Assignment，例如 a2")
    args = ap.parse_args()
    select_gpu()
    print(f"使用 GPU: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '(默认)')}")
    total = mx = 0
    for a in discover(args.lab_dir):
        if args.only and not a.name.startswith(args.only + "-"):
            continue
        r = grade_assignment(a)
        total += r["score"]
        mx += r["max"]
    print(f"Score: {total}/{mx}")


if __name__ == "__main__":
    main()

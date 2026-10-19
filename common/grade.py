#!/usr/bin/env python3
"""判分驱动：编译每个 Assignment，逐个用例在独立进程中运行，汇总分数。仅用标准库。"""
import argparse, json, os, random, re, subprocess, sys
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
        names = "、".join(h.name for h in hips) if hips else "无"
        raise RuntimeError(f"{Path(assign_dir).name} 中应恰好有一个 .hip 文件，实际 {len(hips)} 个（{names}）；"
                           f"请把多余的文件改名为别的后缀（如 .hip.bak）或移出该目录")
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


# 这些变量任意一个已设置，就说明用户（或集群调度器）已经指定了 GPU，判分器不再自动选卡。
VISIBILITY_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
VRAM_FULL = 95.0      # 显存占用 ≥ 此百分比的卡不参与自动选择
BUSY_WARN = 50.0      # 性能用例开始前，所用卡的计算占用超过此百分比就提醒


def _rocm_smi(dev=None):
    """返回 {卡号(str): (计算占用%, 显存占用%)}；rocm-smi 不可用或输出无法解析时返回 {}。"""
    cmd = ["/opt/rocm/bin/rocm-smi", *(["-d", str(dev)] if dev is not None else []),
           "--showuse", "--showmemuse", "--json"]
    try:
        out = subprocess.run(cmd, timeout=30, **_RUN).stdout
        data = json.loads(out[out.index("{"):])
    except Exception:
        return {}
    return parse_smi(data)


def parse_smi(data):
    res = {}
    for card, info in data.items():
        m = re.match(r"card(\d+)$", card)
        if not m:
            continue
        try:
            res[m.group(1)] = (float(info.get("GPU use (%)", 100)),
                               float(info.get("GPU Memory Allocated (VRAM%)", 100)))
        except (ValueError, TypeError, AttributeError):
            continue
    return res


def pick_gpu(usage, rng=random):
    """从 {卡号: (计算%, 显存%)} 中选最空闲的卡：跳过显存 ≥ VRAM_FULL 的卡，按 计算% + 显存% 取最小，并列时随机。"""
    cands = [(u + v, card) for card, (u, v) in usage.items() if v < VRAM_FULL]
    if not cands:
        return None
    best = min(load for load, _ in cands)
    return rng.choice(sorted(card for load, card in cands if load == best))


def select_gpu(env=None, usage=None, rng=random):
    """自动选卡并写入 env["HIP_VISIBLE_DEVICES"]，返回一行中文说明（调用者负责打印）。"""
    env = os.environ if env is None else env
    given = [v for v in VISIBILITY_VARS if v in env]
    if given:
        return "使用 GPU: " + "，".join(f"{v}={env[v]}" for v in given) + "（已由环境变量指定，不自动选卡）"
    usage = _rocm_smi() if usage is None else usage
    card = pick_gpu(usage, rng)
    if card is None:
        if usage:
            return f"使用 GPU: 默认（所有 GPU 显存占用都 ≥ {VRAM_FULL:.0f}%，未自动选卡）"
        return "使用 GPU: 默认（rocm-smi 不可用，未自动选卡）"
    env["HIP_VISIBLE_DEVICES"] = card
    u, v = usage[card]
    return f"使用 GPU: HIP_VISIBLE_DEVICES={card}（自动选择最空闲的卡：计算 {u:.0f}%，显存 {v:.0f}%）"


def current_gpu(env=None):
    """按可见性变量推算当前进程用的物理卡号（尽力而为；无法推算时返回 None）。"""
    env = os.environ if env is None else env
    try:
        devs = None
        rocr = env.get("ROCR_VISIBLE_DEVICES")
        if rocr is not None:
            devs = [int(x) for x in rocr.split(",")]
        for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            if var in env:
                sel = [int(x) for x in env[var].split(",")]
                devs = [devs[i] for i in sel] if devs is not None else sel
                break
        return str(devs[0]) if devs else "0"
    except (ValueError, IndexError):
        return None


def gpu_usage_now():
    dev = current_gpu()
    if dev is None:
        return None, None
    return dev, _rocm_smi(dev).get(dev)


def gpu_load_note():
    """性能用例失败时附上当前所用 GPU 的占用，帮助区分"代码慢"与"别人在用卡"。"""
    dev, u = gpu_usage_now()
    if u is None:
        return ""
    return f"（当前 GPU {dev} 占用：计算 {u[0]:.0f}%，显存 {u[1]:.0f}%）"


def busy_warning():
    """性能用例开始前：所用卡计算占用 > BUSY_WARN 时返回一行提醒，否则返回空串。"""
    dev, u = gpu_usage_now()
    if u is None or u[0] <= BUSY_WARN:
        return ""
    return (f"  注意: GPU {dev} 当前计算占用 {u[0]:.0f}%（可能有别人在用），性能用例的结果可能偏低；"
            f"可用 rocm-smi 查看后换卡（设置 HIP_VISIBLE_DEVICES）重跑")


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
        warned = False
        for c in cases:
            if c["perf"] and not warned and not quiet:  # 第一个性能用例前检查一次所用卡是否正忙
                warned = True
                note = busy_warning()
                if note:
                    print(note)
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
    ap.add_argument("lab_dir", nargs="?")
    ap.add_argument("--only", help="只判某个 Assignment，例如 a2")
    ap.add_argument("--pick-gpu", action="store_true",
                    help="只选卡：说明打印到 stderr；若自动选了卡，把卡号打印到 stdout（供 make run 使用）")
    args = ap.parse_args()
    if args.pick_gpu:
        env = dict(os.environ)
        print(select_gpu(env), file=sys.stderr)
        if "HIP_VISIBLE_DEVICES" in env and "HIP_VISIBLE_DEVICES" not in os.environ:
            print(env["HIP_VISIBLE_DEVICES"])
        return
    if not args.lab_dir:
        ap.error("缺少 lab_dir")
    assigns = [a for a in discover(args.lab_dir) if not args.only or a.name.startswith(args.only + "-")]
    for a in assigns:
        try:
            student_src(a)
        except RuntimeError as e:
            print(f"错误: {e}")
            sys.exit(2)
    print(select_gpu())
    total = mx = 0
    for a in assigns:
        r = grade_assignment(a)
        total += r["score"]
        mx += r["max"]
    print(f"Score: {total}/{mx}")


if __name__ == "__main__":
    main()

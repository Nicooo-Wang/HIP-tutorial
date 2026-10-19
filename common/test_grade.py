import contextlib, io, json, os, random, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade  # noqa: E402

FIX = HERE / "fixtures"
LAB = FIX / "fake-lab"
SOL = FIX / "solutions"


class GradeTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("FAKE_MODE", None)
        os.environ["HK_TIMEOUT"] = "5"

    def test_discover_and_src(self):
        a = grade.discover(LAB)
        self.assertEqual([p.name for p in a], ["a1-fake"])
        self.assertEqual(grade.student_src(a[0]).name, "fake.hip")

    def test_skeleton_scores_partial(self):
        r = grade.grade_assignment(LAB / "a1-fake", quiet=True)
        self.assertTrue(r["built"])
        self.assertEqual(r["max"], 6)
        self.assertEqual(r["score"], 3)  # value_is_42 失败，其余 3 个用例通过

    def test_solution_full_score(self):
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        self.assertEqual(r["score"], r["max"])

    def test_crash_isolated(self):
        os.environ["FAKE_MODE"] = "crash"
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_crash"]["passed"])
        self.assertIn("崩溃", by["maybe_crash"]["msg"])
        self.assertTrue(by["value_is_42"]["passed"])

    def test_hang_times_out(self):
        os.environ["FAKE_MODE"] = "hang"
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_hang"]["passed"])
        self.assertIn("超时", by["maybe_hang"]["msg"])
        self.assertTrue(by["value_is_42"]["passed"])

    def test_perf_retry(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["FAKE_MODE"] = "flaky:" + os.path.join(d, "marker")
            r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
            by = {c["name"]: c for c in r["cases"]}
            self.assertTrue(by["maybe_flaky_perf"]["passed"])
            self.assertEqual(by["maybe_flaky_perf"]["metric"], 2)

    def test_perf_failure_mentions_retries(self):
        os.environ["FAKE_MODE"] = "flaky:/nonexistent-dir/marker"  # 标记文件无法创建 → 每次都失败
        r = grade.grade_assignment(LAB / "a1-fake", src=SOL / "fake-lab/a1-fake/fake.hip", quiet=True)
        by = {c["name"]: c for c in r["cases"]}
        self.assertFalse(by["maybe_flaky_perf"]["passed"])
        self.assertIn("已重试 3 次", by["maybe_flaky_perf"]["msg"])

    def test_build_failure_reported(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "fake.hip"
            bad.write_text("int student_value() { return }\n")
            r = grade.grade_assignment(LAB / "a1-fake", src=bad, quiet=True)
            self.assertFalse(r["built"])
            self.assertEqual(r["score"], 0)
            self.assertIn("error", r["build_log"])

    def test_verify_passes_on_fixture(self):
        p = subprocess.run([sys.executable, str(HERE / "verify.py"), str(LAB), "--solutions-root", str(SOL)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_verify_catches_weak_tests(self):
        # 把 mutant 换成与答案相同的实现：verify 必须报告"mutant 未被抓住"
        with tempfile.TemporaryDirectory() as d:
            sol = Path(d) / "solutions"
            shutil.copytree(SOL, sol)
            (sol / "fake-lab/a1-fake/mutants/m1_wrong.hip").write_text("int student_value() { return 42; }\n")
            p = subprocess.run([sys.executable, str(HERE / "verify.py"), str(LAB), "--solutions-root", str(sol)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn("m1_wrong", p.stdout)


# rocm-smi --showuse --showmemuse --json 的输出样例（字段名与 ROCm 7.2 实测一致）
SMI_JSON = {
    "card0": {"GPU use (%)": "35", "GPU Memory Allocated (VRAM%)": "10"},
    "card1": {"GPU use (%)": "0", "GPU Memory Allocated (VRAM%)": "96"},   # 显存将满：跳过
    "card2": {"GPU use (%)": "0", "GPU Memory Allocated (VRAM%)": "0"},
    "card3": {"GPU use (%)": "0", "GPU Memory Allocated (VRAM%)": "0"},
    "card4": {"GPU use (%)": "N/A", "GPU Memory Allocated (VRAM%)": "0"},  # 无法解析：跳过
    "system": {"Driver version": "x"},
}


class SelectGpuTest(unittest.TestCase):
    """不需要 GPU：用假的 rocm-smi 输出与环境变量字典测试选卡逻辑。"""

    def test_parse_smi(self):
        u = grade.parse_smi(SMI_JSON)
        self.assertEqual(sorted(u), ["0", "1", "2", "3"])
        self.assertEqual(u["0"], (35.0, 10.0))

    def test_skip_full_vram_and_ties_random(self):
        u = grade.parse_smi(SMI_JSON)
        picks = {grade.pick_gpu(u, random.Random(seed)) for seed in range(50)}
        self.assertEqual(picks, {"2", "3"})  # 1 号显存 ≥95% 被跳过；2、3 并列最空闲，随机选

    def test_lowest_load_wins(self):
        u = {"0": (10.0, 0.0), "1": (0.0, 5.0), "2": (0.0, 94.0)}
        self.assertEqual(grade.pick_gpu(u, random.Random(0)), "1")

    def test_all_full_or_no_smi(self):
        env = {}
        msg = grade.select_gpu(env, usage={"0": (0.0, 95.0), "1": (0.0, 99.0)})
        self.assertNotIn("HIP_VISIBLE_DEVICES", env)
        self.assertIn("95%", msg)
        env = {}
        msg = grade.select_gpu(env, usage={})
        self.assertNotIn("HIP_VISIBLE_DEVICES", env)
        self.assertIn("rocm-smi", msg)

    def test_auto_select_sets_env(self):
        env = {}
        msg = grade.select_gpu(env, usage=grade.parse_smi(SMI_JSON), rng=random.Random(1))
        self.assertIn(env["HIP_VISIBLE_DEVICES"], ("2", "3"))
        self.assertIn("自动选择", msg)

    def test_visibility_vars_honoured(self):
        for var in grade.VISIBILITY_VARS:
            env = {var: "5"}
            msg = grade.select_gpu(env, usage=grade.parse_smi(SMI_JSON))
            self.assertEqual(env, {var: "5"}, var)  # 不改动、不额外设置 HIP_VISIBLE_DEVICES
            self.assertIn(f"{var}=5", msg)
            self.assertIn("不自动选卡", msg)

    def test_current_gpu(self):
        self.assertEqual(grade.current_gpu({}), "0")
        self.assertEqual(grade.current_gpu({"HIP_VISIBLE_DEVICES": "6,7"}), "6")
        self.assertEqual(grade.current_gpu({"CUDA_VISIBLE_DEVICES": "3"}), "3")
        self.assertEqual(grade.current_gpu({"GPU_DEVICE_ORDINAL": "4"}), "4")
        self.assertEqual(grade.current_gpu({"ROCR_VISIBLE_DEVICES": "4,5"}), "4")
        self.assertEqual(grade.current_gpu({"ROCR_VISIBLE_DEVICES": "4,5", "HIP_VISIBLE_DEVICES": "1"}), "5")
        self.assertIsNone(grade.current_gpu({"HIP_VISIBLE_DEVICES": "GPU-abcdef"}))

    def test_busy_warning(self):
        orig, env_orig = grade._rocm_smi, os.environ.get("HIP_VISIBLE_DEVICES")
        try:
            os.environ["HIP_VISIBLE_DEVICES"] = "2"
            grade._rocm_smi = lambda dev=None: {"2": (80.0, 30.0)}
            self.assertIn("80%", grade.busy_warning())
            self.assertIn("80%", grade.gpu_load_note())
            grade._rocm_smi = lambda dev=None: {"2": (50.0, 30.0)}
            self.assertEqual(grade.busy_warning(), "")
            grade._rocm_smi = lambda dev=None: {}
            self.assertEqual(grade.busy_warning(), "")
            self.assertEqual(grade.gpu_load_note(), "")
        finally:
            grade._rocm_smi = orig
            if env_orig is None:
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
            else:
                os.environ["HIP_VISIBLE_DEVICES"] = env_orig

    def test_pick_gpu_cli_respects_env(self):
        env = {**os.environ, "ROCR_VISIBLE_DEVICES": "3"}
        p = subprocess.run([sys.executable, str(HERE / "grade.py"), "--pick-gpu"], env=env,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(p.stdout, "")  # 已指定：不输出卡号，make run 就不会覆盖
        self.assertIn("ROCR_VISIBLE_DEVICES=3", p.stderr)


class TwoHipFilesTest(unittest.TestCase):
    """Assignment 目录里有两个 .hip：grade.py 与 make run 都打印一行中文并以退出码 2 结束（不编译、不需要 GPU）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lab = Path(self.tmp.name) / "fake-lab"
        shutil.copytree(LAB / "a1-fake", self.lab / "a1-fake")
        (self.lab / "a1-fake" / "fake_copy.hip").write_text("int student_value() { return 42; }\n")
        (self.lab / "Makefile").write_text(f"include {HERE / 'lab.mk'}\n")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, p):
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        out = (p.stdout + p.stderr).strip()
        self.assertIn("应恰好有一个 .hip 文件，实际 2 个", out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("error:", out)

    def test_grade(self):
        self.check(subprocess.run([sys.executable, str(HERE / "grade.py"), str(self.lab)],
                                  capture_output=True, text=True))

    def test_make_run(self):
        p = subprocess.run(["make", "-s", "-C", str(self.lab), "run", "A=a1"], capture_output=True, text=True)
        self.check(p)
        self.assertFalse((self.lab / "build").exists())  # 没有去编译


if __name__ == "__main__":
    unittest.main()

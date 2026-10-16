import os, shutil, subprocess, sys, tempfile, unittest
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


if __name__ == "__main__":
    unittest.main()

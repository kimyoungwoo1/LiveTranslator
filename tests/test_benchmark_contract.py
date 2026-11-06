"""Offline validation for benchmark inputs and output report formatting."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


class BenchmarkContractTests(unittest.TestCase):
    def test_models_are_not_silently_added_to_live_executable(self):
        for path in ("compare_translation_models.py", "report.py"):
            ast.parse((BENCH / path).read_text(encoding="utf-8"))
        self.assertNotIn("quickmt", (ROOT / "requirements.txt").read_text(encoding="utf-8"))
        self.assertIn('value="hymt"', (ROOT / "app_core.py").read_text(encoding="utf-8"))

    def test_clean_reference_and_noisy_asr_cases_are_separate(self):
        cases = json.loads((BENCH / "en_ko_cases.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases["clean"]), 12)
        self.assertGreaterEqual(len(cases["noisy"]), 4)
        self.assertEqual(
            len({case["id"] for case in cases["clean"] + cases["noisy"]}),
            len(cases["clean"] + cases["noisy"]),
        )
        self.assertTrue(all(case.get("ko") for case in cases["clean"]))
        self.assertTrue(all("ko" not in case for case in cases["noisy"]))
        self.assertTrue(any("60 days" in case["en"] for case in cases["clean"]))

    def test_report_marks_failed_inference_honestly(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("report_tested", BENCH / "report.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(report, "RESULTS", Path(folder)):
                rendered = report.render_report()
        self.assertIn("failed/incomplete", rendered)
        self.assertIn("No winner is declared automatically", rendered)
        self.assertIn("QuickMT", rendered)
        self.assertIn("Hy-MT2", rendered)


if __name__ == "__main__":
    unittest.main()

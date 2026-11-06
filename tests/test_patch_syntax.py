"""Regression: never publish another syntactically invalid in-app Python patch."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PatchSyntaxTests(unittest.TestCase):
    def test_every_current_update_source_compiles(self):
        for p in (ROOT / "app_core.py", ROOT / "streaming_pipeline.py",
                  ROOT / "updates/v0.6.19/app_core.py",
                  ROOT / "updates/v0.6.19/streaming_pipeline.py"):
            with self.subTest(path=str(p)):
                compile(p.read_bytes(), str(p), "exec")

    def test_english_terminal_regex_imports_and_handles_punctuation(self):
        path = ROOT / "streaming_pipeline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "SentenceBuffer")
        pattern_assign = next(n for n in cls.body
                              if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "TERMINAL"
                                      for t in n.targets))
        expr = ast.Expression(body=pattern_assign.value)
        regex = eval(compile(expr, str(path), "eval"), {"re": re})
        self.assertTrue(regex.search("The new order arrived."))
        self.assertTrue(regex.search("Is the payment due?"))
        self.assertFalse(regex.search("We have another new order"))
        self.assertFalse(regex.search("new...".removesuffix("...")))


if __name__ == "__main__":
    unittest.main()

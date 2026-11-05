"""Contract tests for deterministic Argos quantity safeguards (no AI dependencies)."""
import ast
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import unittest


def extract_guard():
    path = Path(__file__).resolve().parent.parent / "app_core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_SCALE", "_ENGLISH_NUMBERS", "_SCALE_EXPR", "_KOREAN_SCALE",
             "_format_korean_large_number", "guard_argos_quantity"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and (
            (isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name in names)
        )
    ]
    compiled = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    namespace = {"re": re, "Decimal": Decimal, "InvalidOperation": InvalidOperation}
    exec(compiled, namespace)
    return namespace["guard_argos_quantity"]


class ArgosQuantityGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = staticmethod(extract_guard())

    def test_six_billion_euro_is_sixty_eok(self):
        out = self.guard(
            "Volkswagen Group lost nearly six billion euro.",
            "폭스바겐 그룹은 거의 6억 유로를 잃었다."
        )
        self.assertEqual(out, "폭스바겐 그룹은 거의 60억 유로를 잃었다.")

    def test_correct_magnitude_is_untouched(self):
        original = "거의 60억 유로 손실"
        self.assertEqual(self.guard("lost six billion euro", original), original)

    def test_numeric_and_decimal_billion(self):
        self.assertEqual(self.guard("lost 1.2 billion euros", "12억 유로 손실"), "12억 유로 손실")
        self.assertEqual(self.guard("lost 1.2 billion euros", "1억 유로 손실"), "12억 유로 손실")

    def test_million(self):
        self.assertEqual(self.guard("lost six million euros", "6만 유로 손실"), "600만 유로 손실")

    def test_multiple_quantities_are_not_rewritten(self):
        out = "매출 6억, 이익 3억"
        self.assertEqual(self.guard("six billion revenue and three billion profit", out), out)

    def test_compound_korean_amount_is_not_rewritten(self):
        out = "손실 2억 5천만 유로"
        self.assertEqual(self.guard("loss of 2.5 billion euros", out), out)

    def test_unscaled_english_numbers_do_not_change_korean(self):
        self.assertEqual(self.guard("six percent", "6억"), "6억")

    def test_unmatched_english_word_or_korean_magnitude_stays(self):
        self.assertEqual(self.guard("about one hundred billion euros", "100억 유로"), "100억 유로")


if __name__ == "__main__":
    unittest.main()

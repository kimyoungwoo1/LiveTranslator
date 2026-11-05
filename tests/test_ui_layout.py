"""Validate the redesigned Tkinter dashboard without requiring Windows GUI display."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "app_core.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC, filename=str(ROOT / "app_core.py"))


def app_method(method):
    cls = next(node for node in TREE.body
               if isinstance(node, ast.ClassDef) and node.name == "App")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == method)


class DashboardTests(unittest.TestCase):
    def test_ui_keeps_core_actions_and_explicit_exit(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        for item in (
            'text="프로그램 종료", command=self.close',
            'text="▶  번역 시작", command=self.start',
            'text="■  중지", command=self.stop',
            'text="자막 지우기", command=self.clear_output',
            'text="소리 찾기", command=self.scan_audio',
            'text="업데이트 확인", command=self.check_update',
            'command=self.diagnose_models',
            'command=self.configure_google_key',
            'self.remember_selected_device',
            'self.install_pack("en")',
            'self.install_pack("ja")',
        ):
            with self.subTest(item=item):
                self.assertIn(item, ui)

    def test_both_transcript_boxes_and_overlay_are_preserved(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertIn('transcript_pane("원문', ui)
        self.assertIn('transcript_pane("한국어 번역', ui)
        self.assertIn('orig_pane, self.original = transcript_pane', ui)
        self.assertIn('ko_pane, self.translated = transcript_pane', ui)
        self.assertIn('transcript.add(orig_pane, weight=1)', ui)
        self.assertIn('transcript.add(ko_pane, weight=1)', ui)
        self.assertIn('self.overlay_text = tk.Label(', ui)

    def test_exit_confirms_active_translation_and_signals_stop(self):
        method = ast.get_source_segment(SRC, app_method("close"))
        self.assertIn("if self.running and not messagebox.askyesno", method)
        self.assertIn("self.stop_signal.set()", method)
        self.assertIn("self.root.destroy()", method)

    def test_status_indicator_is_refreshed_on_start_stop_end(self):
        for method in ("start", "stop", "consume"):
            code = ast.get_source_segment(SRC, app_method(method))
            self.assertIn("self.set_live_indicator(", code)

    def test_no_new_gui_or_external_dependencies(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertIn('style.theme_use("clam")', ui)
        self.assertNotIn("customtkinter", SRC)
        self.assertNotIn("PySide", SRC)


if __name__ == "__main__":
    unittest.main()

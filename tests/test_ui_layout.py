"""Validate the redesigned Tkinter dashboard without requiring Windows GUI display."""
import ast
import os
import sys
from pathlib import Path
import unittest
from unittest import mock

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

    def test_footer_is_reserved_before_expanding_transcripts(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertLess(ui.index('footer.pack(side="bottom", fill="x")'),
                        ui.index('transcript.pack(fill="both", expand=True'))
        self.assertIn('min(1050, screen_h - 105)', ui)
        self.assertIn('height=9, wrap="word"', ui)

    def test_hymt_is_recommended_first_and_argos_not_mislabeled(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertLess(ui.index('text="Hy-MT2 · 기본 추천'),
                        ui.index('text="Google Cloud · 온라인'))
        self.assertLess(ui.index('text="Google Cloud · 온라인'),
                        ui.index('text="Argos · 경량 오프라인'))
        self.assertNotIn('Argos 고품질', ui)
        self.assertIn('self.engine_tools_frame = tools', ui)
        initial = ast.get_source_segment(SRC, app_method("__init__"))
        self.assertIn('self.settings.get("engine", "hymt")', initial)

    def test_model_loading_status_is_not_silent_and_marks_completion(self):
        worker = ast.get_source_segment(SRC, app_method("capture_and_translate"))
        ui = ast.get_source_segment(SRC, app_method("consume"))
        self.assertIn('("model_loading", (model_name, path == model_name))', worker)
        self.assertIn('("model_loaded", model_name)', worker)
        self.assertIn('elif kind == "model_loading":', ui)
        self.assertIn('elif kind == "model_loaded":', ui)
        self.assertIn('seconds_waiting = int(time.monotonic() - started)', ui)
        self.assertIn('최초 다운로드', ui)
        self.assertIn('self._model_loading_started = None', ui)

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

    def test_oriented_widgets_use_builtin_ttk_layouts(self):
        """Regression: custom scrollbar styles cause missing Vertical.* layouts."""
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertIn('ttk.Scrollbar(text_row, orient="vertical"', ui)
        self.assertIn('ttk.Panedwindow(outer, orient="vertical")', ui)
        self.assertIn('ttk.Progressbar(meter, variable=self.audio_meter', ui)
        for broken_style in (
            'style="LT.TScrollbar"',
            'style="LT.Horizontal.TProgressbar"',
            'style="LT.TPanedwindow"',
        ):
            self.assertNotIn(broken_style, ui)

    @unittest.skipUnless(os.name == "nt", "Real Tk widget smoke test runs on Windows")
    def test_windows_real_tk_window_opens_without_scrollbar_tclerror(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Windows runner has no interactive desktop: {exc}")
        root.withdraw()
        try:
            with mock.patch.dict(sys.modules, {
                "numpy": mock.MagicMock(),
                "scipy": mock.MagicMock(),
                "scipy.signal": mock.MagicMock(),
                "pyaudiowpatch": mock.MagicMock(),
            }):
                import app_core
            with mock.patch.object(app_core.App, "refresh"):
                app = app_core.App(root)
                root.update_idletasks()
            self.assertIsNotNone(app.original)
            self.assertIsNotNone(app.translated)
        finally:
            root.destroy()

    def test_no_new_gui_or_external_dependencies(self):
        ui = ast.get_source_segment(SRC, app_method("build_ui"))
        self.assertIn('style.theme_use("clam")', ui)
        self.assertNotIn("customtkinter", SRC)
        self.assertNotIn("PySide", SRC)


if __name__ == "__main__":
    unittest.main()

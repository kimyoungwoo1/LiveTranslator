"""One paired transcript plus one explicitly marked AI-summary TXT per saved session."""
import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


def load_app():
    with mock.patch.dict(sys.modules, {
        "numpy": mock.MagicMock(),
        "scipy": mock.MagicMock(),
        "scipy.signal": mock.MagicMock(),
        "pyaudiowpatch": mock.MagicMock(),
    }):
        import app_core
    return app_core


class TranscriptFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_app()

    def app(self, engine="hymt", enabled=True):
        app = self.module.App.__new__(self.module.App)
        app.save_enabled = mock.MagicMock()
        app.save_enabled.get.return_value = enabled
        app.status = mock.MagicMock()
        app.session_started_at = datetime.datetime(2026, 11, 6, 12, 31, 25)
        app.session_record_path = None
        app.session_summary_path = None
        app.session_ko_lines = []
        app.session_engine = engine
        app.session_language = "en"
        return app

    def test_original_and_korean_are_paired_in_one_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app()
                app.save_transcript("New order. 60 days.", "새 주문. 60일.")
                path = app.session_record_path
                self.assertEqual(path.name, "2026-11-06_12-31-25_Hy-MT2_en.txt")
                self.assertEqual(path.parent, Path(folder))
                self.assertEqual(len(list(Path(folder).glob("*.txt"))), 2)
                result = path.read_text(encoding="utf-8-sig")
                self.assertIn("번역 엔진: Hy-MT2", result)
                self.assertIn("[원문]\nNew order. 60 days.\n[한국어 번역]\n새 주문. 60일.", result)

    def test_multiple_translation_pairs_remain_separated(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app()
                app.save_transcript("One order.", "주문 한 건.")
                app.save_transcript("Pay in 60 days.", "60일 이내에 결제하세요.")
                result = app.session_record_path.read_text(encoding="utf-8-sig")
                self.assertEqual(result.count("[원문]"), 2)
                self.assertEqual(result.count("[한국어 번역]"), 2)
                self.assertLess(result.index("One order."), result.index("주문 한 건."))
                self.assertLess(result.index("주문 한 건."), result.index("Pay in 60 days."))
                self.assertLess(result.index("Pay in 60 days."), result.index("60일 이내에 결제하세요."))
                self.assertEqual(len(list(Path(folder).glob("*.txt"))), 2)

    def test_sessions_have_unique_filenames_even_same_second(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                first = self.app()
                first.save_transcript("First call", "첫 통화")
                second = self.app()
                second.save_transcript("Second call", "두 번째 통화")
                self.assertNotEqual(first.session_record_path, second.session_record_path)
                self.assertEqual(second.session_record_path.stem,
                                 first.session_record_path.stem + "_01")
                self.assertIn("First call", first.session_record_path.read_text(encoding="utf-8-sig"))
                self.assertIn("Second call", second.session_record_path.read_text(encoding="utf-8-sig"))

    def test_existing_older_records_and_folders_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "2026-11-06.txt"
            old.write_text("historic translation\n", encoding="utf-8")
            old_session = root / "2026-11-06_12-31-25_Hy-MT2_en"
            old_session.mkdir()
            (old_session / "원문.txt").write_text("prior source\n", encoding="utf-8")
            (old_session / "한국어_번역.txt").write_text("prior translation\n", encoding="utf-8")
            with mock.patch.object(self.module, "TRANSCRIPTS", root):
                self.app().save_transcript("New order.", "새 주문.")
            self.assertEqual(old.read_text(encoding="utf-8"), "historic translation\n")
            self.assertEqual((old_session / "원문.txt").read_text(encoding="utf-8"), "prior source\n")
            self.assertEqual((old_session / "한국어_번역.txt").read_text(encoding="utf-8"), "prior translation\n")

    def test_disabled_recording_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app(enabled=False)
                app.save_transcript("Do not save", "저장하지 않음")
                self.assertIsNone(app.session_record_path)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_second_file_uses_same_session_stem_and_waits_for_ai(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app()
                app.save_transcript("Can you extend the terms?", "지불 기한을 연장할 수 있나요?")
                self.assertEqual(
                    app.session_summary_path.name,
                    "2026-11-06_12-31-25_Hy-MT2_en_AI요약.txt",
                )
                self.assertEqual(
                    app.session_ko_lines, ["지불 기한을 연장할 수 있나요?"]
                )
                self.assertIn("생성 대기", app.session_summary_path.read_text(encoding="utf-8-sig"))

    def test_ai_summary_chunker_preserves_all_text_in_order(self):
        chunk = self.module.App._summary_chunks
        text = ["A " * 16, "B " * 17, "C " * 19]
        parts = chunk(text, max_chars=32)
        self.assertTrue(len(parts) > 1)
        self.assertEqual(" ".join(" ".join(parts).split()), " ".join(" ".join(text).split()))
        self.assertTrue(all(len(part) <= 32 for part in parts))

    def test_summary_job_not_launched_without_saved_session(self):
        app = self.app(enabled=False)
        app.summary_running = False
        app.settings = {}
        with mock.patch.object(self.module.threading, "Thread") as thread:
            app._start_ai_summary()
        thread.assert_not_called()

    def test_hymt_summary_requires_installed_local_model(self):
        app = self.app()
        with tempfile.TemporaryDirectory() as folder:
            summary_path = Path(folder) / "test_AI요약.txt"
            # Import via patched optional audio dependencies, just as the app does.
            with mock.patch.dict(sys.modules, {
                "numpy": mock.MagicMock(),
                "scipy": mock.MagicMock(),
                "scipy.signal": mock.MagicMock(),
                "pyaudiowpatch": mock.MagicMock(),
            }):
                import streaming_pipeline
                with mock.patch.object(streaming_pipeline, "hym_ready", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "설치되지 않아"):
                        app._create_ai_summary(summary_path, ["안녕하세요"], None)

    def test_windows_explorer_still_opens_records_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            opened = mock.MagicMock()
            fake_os = SimpleNamespace(name="nt", startfile=opened)
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)), \
                 mock.patch.object(self.module, "os", fake_os):
                self.app().open_transcripts_folder()
            opened.assert_called_once_with(str(Path(folder)))

    def test_ui_saves_paired_result_not_original_event(self):
        source = (Path(__file__).resolve().parents[1] / "app_core.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.save_transcript(original, translated)", source)
        self.assertNotIn('self.save_transcript("original", data)', source)
        self.assertIn("command=self.open_transcripts_folder", source)
        self.assertIn('원문·번역 + AI 요약 TXT 2개 저장', source)
        self.assertNotIn('datetime.date.today().isoformat() + ".txt"', source)


if __name__ == "__main__":
    unittest.main()

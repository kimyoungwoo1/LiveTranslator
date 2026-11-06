"""Records are separated by launch session and source/translation text."""
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
        app.session_record_dir = None
        app.session_engine = engine
        app.session_language = "en"
        return app

    def test_original_and_korean_are_in_separate_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app()
                app.save_transcript("original", "New order. 60 days.")
                app.save_transcript("translation", "새 주문. 60일.")
                path = app.session_record_dir
                self.assertEqual(path.name, "2026-11-06_12-31-25_Hy-MT2_en")
                original = (path / "원문.txt").read_text(encoding="utf-8-sig")
                korean = (path / "한국어_번역.txt").read_text(encoding="utf-8-sig")
                self.assertIn("New order. 60 days.", original)
                self.assertNotIn("새 주문. 60일.", original)
                self.assertIn("새 주문. 60일.", korean)
                self.assertNotIn("New order. 60 days.", korean)
                self.assertIn("Hy-MT2", original)
                self.assertIn("Hy-MT2", korean)

    def test_sessions_are_unique_even_with_same_start_second(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                first = self.app()
                first.save_transcript("original", "First call")
                second = self.app()
                second.save_transcript("original", "Second call")
                self.assertNotEqual(first.session_record_dir, second.session_record_dir)
                self.assertEqual(second.session_record_dir.name,
                                 first.session_record_dir.name + "_01")
                self.assertIn("First call", (first.session_record_dir / "원문.txt")
                              .read_text(encoding="utf-8-sig"))
                self.assertIn("Second call", (second.session_record_dir / "원문.txt")
                              .read_text(encoding="utf-8-sig"))

    def test_existing_daily_text_is_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "2026-11-06.txt"
            old.write_text("historic translation\n", encoding="utf-8")
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app()
                app.save_transcript("translation", "오늘 새 번역")
            self.assertEqual(old.read_text(encoding="utf-8"), "historic translation\n")

    def test_saving_disabled_does_not_create_record(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)):
                app = self.app(enabled=False)
                app.save_transcript("original", "Do not save")
                self.assertIsNone(app.session_record_dir)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_open_folder_uses_windows_explorer(self):
        with tempfile.TemporaryDirectory() as folder:
            opened = mock.MagicMock()
            fake_os = SimpleNamespace(name="nt", startfile=opened)
            with mock.patch.object(self.module, "TRANSCRIPTS", Path(folder)), \
                 mock.patch.object(self.module, "os", fake_os):
                self.app().open_transcripts_folder()
            opened.assert_called_once_with(str(Path(folder)))

    def test_ui_events_write_to_separate_files(self):
        source = (Path(__file__).resolve().parents[1] / "app_core.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('self.save_transcript("original", data)', source)
        self.assertIn('self.save_transcript("translation", translated)', source)
        self.assertIn("command=self.open_transcripts_folder", source)
        self.assertNotIn('datetime.date.today().isoformat() + ".txt"', source)


if __name__ == "__main__":
    unittest.main()

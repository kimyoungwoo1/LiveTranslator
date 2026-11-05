"""Quality-mode sentence assembly regression tests (no live audio required)."""
import importlib.util
from pathlib import Path
import re
import sys
import unittest
from unittest import mock


def load_pipeline():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("quality_pipeline_tested", root / "streaming_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "numpy": mock.MagicMock(),
        "scipy": mock.MagicMock(),
        "scipy.signal": mock.MagicMock(),
        "pyaudiowpatch": mock.MagicMock(),
    }):
        spec.loader.exec_module(module)
    return module


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()

    def test_english_fragments_form_one_sentence(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("We should", "en", 0), [])
        self.assertEqual(buf.add("review the configuration", "en", 3), [])
        self.assertEqual(buf.add("before deployment.", "en", 6), [
            ("We should review the configuration before deployment.", "en")
        ])

    def test_japanese_fragments_do_not_add_english_spaces(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("この設定を", "ja", 0), [])
        self.assertEqual(buf.add("確認します。", "ja", 3), [
            ("この設定を確認します。", "ja")
        ])

    def test_max_fragments_flushes_when_punctuation_missing(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("first", "en", 0), [])
        self.assertEqual(buf.add("second", "en", 3), [])
        self.assertEqual(buf.add("third", "en", 6), [("first second third", "en")])

    def test_language_change_does_not_mix_translation_languages(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("Hello", "en", 0), [])
        self.assertEqual(buf.add("こんにちは。", "ja", 3), [
            ("Hello", "en"), ("こんにちは。", "ja")
        ])

    def test_immediate_duplicate_is_not_appended_twice(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("Hello", "en", 0), [])
        self.assertEqual(buf.add("Hello", "en", 3), [])
        self.assertEqual(buf.add("world.", "en", 6), [("Hello world.", "en")])

    def test_silence_flush_preserves_incomplete_sentence(self):
        buf = self.pipeline.SentenceBuffer(3)
        buf.add("partial phrase", "en", 0)
        self.assertEqual(buf.flush(), ("partial phrase", "en"))
        self.assertIsNone(buf.flush())

    def test_argos_one_quality_profile_and_google_still_selectable(self):
        app = (Path(__file__).resolve().parent.parent / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["ARGOS_BEAM_SIZE"] = "4"', app)
        self.assertIn('os.environ["ARGOS_COMPUTE_TYPE"] = "float32"', app)
        self.assertIn('quality_mode=(engine == "argos")', app)
        self.assertIn('if engine == "google":', app)
        self.assertNotIn('mode="balanced"', app)


if __name__ == "__main__":
    unittest.main()

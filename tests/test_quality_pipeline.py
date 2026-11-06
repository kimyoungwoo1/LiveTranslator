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

    def test_ellipsis_is_incomplete_and_next_chunk_completes_order(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("Actually, I also want to place another new...", "en", 0), []
        )
        self.assertEqual(
            buf.add("order. The same size as last time.", "en", 4),
            [("Actually, I also want to place another new order. The same size as last time.", "en")]
        )

    def test_repeated_audio_prefix_is_not_repeated_in_transcript(self):
        dedup = self.pipeline.strip_transcript_overlap
        self.assertEqual(
            dedup(
                "We can make an exception this time.",
                "an exception this time. That's great, Junko.",
            ),
            "That's great, Junko."
        )
        self.assertEqual(
            dedup(
                "Please extend the payment terms to 60 days.",
                "to 60 days. Thanks for your help.",
            ),
            "Thanks for your help."
        )

    def test_genuine_repeated_words_survive(self):
        dedup = self.pipeline.strip_transcript_overlap
        self.assertEqual(dedup("Yes.", "Yes, yes.", "en"), "Yes, yes.")
        self.assertEqual(dedup("", "Hello.", "en"), "Hello.")
        self.assertEqual(
            dedup("こんにちは。", "こんにちは。", "ja"), "こんにちは。"
        )

    def test_no_unsafe_overlap_dedup_without_adjacent_match(self):
        dedup = self.pipeline.strip_transcript_overlap
        self.assertEqual(
            dedup("I would like to make another order.",
                  "A new order should be sent today."),
            "A new order should be sent today."
        )

    def test_overlapping_audio_is_used_only_in_quality_mode(self):
        source = (Path(__file__).resolve().parent.parent / "streaming_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("QUALITY_OVERLAP_SECONDS = 0.7", source)
        self.assertIn("block, overlap_tail = add_audio_context(", source)
        self.assertIn("last_recognized_audio_text = raw_original", source)
        self.assertIn("strip_transcript_overlap(", source)
        self.assertIn("if not quality_mode or block.size == 0:", source)

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

    def test_beam_size_adapts_to_cpu_backlog(self):
        decide = self.pipeline.choose_asr_beam_size
        self.assertEqual(decide(True, 0), 5)
        self.assertEqual(decide(True, 1), 5)
        self.assertEqual(decide(True, 2), 3)
        self.assertEqual(decide(True, 3), 3)
        self.assertEqual(decide(True, 4), 2)
        self.assertEqual(decide(False, 4), 1)

    def test_pipeline_is_not_accidentally_duplicated(self):
        source = (Path(__file__).resolve().parent.parent / "streaming_pipeline.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("def run_stream("), 1)
        self.assertEqual(source.count("def resolve_capture_device("), 1)

    def test_quality_waits_for_pauses_after_minimum_context(self):
        ready = self.pipeline.should_finalize_audio
        rate = 16000
        self.assertFalse(ready(int(2.9 * rate), int(0.5 * rate), rate, 3, True))
        self.assertFalse(ready(int(3.2 * rate), int(0.15 * rate), rate, 3, True))
        self.assertTrue(ready(int(3.3 * rate), int(0.32 * rate), rate, 3, True))

    def test_quality_forces_cut_at_bounded_maximum(self):
        ready = self.pipeline.should_finalize_audio
        rate = 48000
        self.assertFalse(ready(int(4.8 * rate), 0, rate, 3, True))
        self.assertTrue(ready(int(5.1 * rate), 0, rate, 3, True))
        self.assertTrue(ready(int(10 * rate), 0, rate, 8, True))

    def test_google_keeps_original_fixed_interval(self):
        ready = self.pipeline.should_finalize_audio
        rate = 48000
        self.assertFalse(ready(int(2.9 * rate), 0, rate, 3, False))
        self.assertTrue(ready(int(3 * rate), 0, rate, 3, False))

    def test_quality_asr_uses_beam_five_and_bounded_context(self):
        path = Path(__file__).resolve().parent.parent / "streaming_pipeline.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn("beam_size=choose_asr_beam_size(quality_mode, pending.qsize())", text)
        self.assertIn("initial_prompt=prompt", text)
        self.assertIn("recent_transcript[-140:]", text)
        self.assertIn('recent_transcript = ""', text)

    def test_translation_runs_in_separate_worker_not_asr_loop(self):
        import ast
        root = Path(__file__).resolve().parent.parent
        program = ast.parse((root / "streaming_pipeline.py").read_text(encoding="utf-8"))
        stream = next(node for node in program.body
                      if isinstance(node, ast.FunctionDef) and node.name == "run_stream")
        methods = {node.name: node for node in stream.body
                   if isinstance(node, ast.FunctionDef)}
        self.assertIn("deliver", methods)
        self.assertIn("translation_worker", methods)
        deliver = ast.unparse(methods["deliver"])
        translate_worker = ast.unparse(methods["translation_worker"])
        self.assertIn("translation_pending.put_nowait(task)", deliver)
        self.assertNotIn("translate(", deliver)
        self.assertIn("translate(text, source_lang)", translate_worker)
        self.assertIn("emit_result(text, result)", translate_worker)
        self.assertIn("translation_thread.start()", ast.unparse(stream))
        self.assertIn("음성 누락", ast.unparse(stream))
        self.assertIn("번역 누락", ast.unparse(stream))

    def test_translation_thread_reports_errors_to_ui(self):
        app = (Path(__file__).resolve().parent.parent / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('emit_error=lambda message: self.events.put(("error", message))', app)

    def test_argos_one_quality_profile_and_google_still_selectable(self):
        app = (Path(__file__).resolve().parent.parent / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["ARGOS_BEAM_SIZE"] = "4"', app)
        self.assertIn('os.environ["ARGOS_COMPUTE_TYPE"] = "float32"', app)
        self.assertIn('quality_mode=(engine in ("argos", "hymt"))', app)
        self.assertIn('if engine == "google":', app)
        self.assertNotIn('mode="balanced"', app)


if __name__ == "__main__":
    unittest.main()

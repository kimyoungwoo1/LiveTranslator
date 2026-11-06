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
        self.assertIn("QUALITY_OVERLAP_SECONDS = 0.9", source)
        self.assertIn("block, overlap_tail = add_audio_context(", source)
        self.assertIn("last_recognized_audio_text = cleaned_original", source)
        self.assertIn("strip_transcript_overlap(", source)
        self.assertIn("if not quality_mode or block.size == 0:", source)

    def test_audio_context_keeps_900ms_tail_once(self):
        class Samples(list):
            @property
            def size(self):
                return len(self)
            def copy(self):
                return Samples(self)
            def __getitem__(self, key):
                result = super().__getitem__(key)
                return Samples(result) if isinstance(key, slice) else result

        concat = lambda arrays: Samples(list(arrays[0]) + list(arrays[1]))
        with mock.patch.object(self.pipeline.np, "concatenate", side_effect=concat):
            first, tail = self.pipeline.add_audio_context(
                Samples([1, 2, 3, 4]), Samples([]), rate=4
            )
            self.assertEqual(first, [1, 2, 3, 4])
            self.assertEqual(tail, [2, 3, 4])  # int(4 * 0.9) = 3
            second, tail2 = self.pipeline.add_audio_context(
                Samples([5, 6, 7, 8]), tail, rate=4
            )
            self.assertEqual(second, [2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(tail2, [6, 7, 8])
            self.assertEqual(first + Samples([5, 6, 7, 8]),
                             [1, 2, 3, 4, 5, 6, 7, 8])

    def test_plain_google_mode_does_not_duplicate_audio(self):
        class Samples(list):
            @property
            def size(self):
                return len(self)

        with mock.patch.object(self.pipeline.np, "empty", return_value=Samples([])):
            chunk, tail = self.pipeline.add_audio_context(
                Samples([1, 2, 3]), Samples([9, 10]), rate=4,
                quality_mode=False,
            )
        self.assertEqual(chunk, [1, 2, 3])
        self.assertEqual(tail, [])

    def test_split_let_me_see_is_reassembled_for_translation(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("That sounds good. Hold on, Andrea. Let me see.", "en", 0),
            [("That sounds good. Hold on, Andrea.", "en")]
        )
        self.assertEqual(
            buf.add("what I can do. Yes.", "en", 11),
            [("Let me see what I can do. Yes.", "en")]
        )

    def test_split_it_works_both_ways_is_one_translation(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("And we appreciate your business, Andrea. It works.", "en", 0),
            [("And we appreciate your business, Andrea.", "en")]
        )
        self.assertEqual(
            buf.add("both ways. Thanks again, Junko.", "en", 9),
            [("It works both ways. Thanks again, Junko.", "en")]
        )

    def test_british_council_negation_survives_chunk_boundary(self):
        buf = self.pipeline.SentenceBuffer(3)
        first = (
            "You'd really be helping us out if you could extend it "
            "to 60 days? I'm not."
        )
        self.assertEqual(
            buf.add(first, "en", 0),
            [("You'd really be helping us out if you could extend it "
              "to 60 days?", "en")]
        )
        self.assertEqual(
            buf.add("sure if I can do that, Andrea. We've got regulations at our end.",
                    "en", 5),
            [("I'm not sure if I can do that, Andrea. "
              "We've got regulations at our end.", "en")]
        )

    def test_british_council_negation_does_not_absorb_new_speaker(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("I'm not.", "en", 0), [])
        self.assertEqual(
            buf.add("Everything is fine with the order.", "en", 4),
            [("I'm not.", "en"), ("Everything is fine with the order.", "en")]
        )

    def test_explicit_negative_full_sentence_is_not_held(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("I'm not sure if we can extend the payment terms.", "en", 0),
            [("I'm not sure if we can extend the payment terms.", "en")]
        )

    def test_completed_questions_translate_immediately(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("Can you send me a quick email confirmation?", "en", 0),
            [("Can you send me a quick email confirmation?", "en")]
        )

    def test_new_utterance_does_not_merge_with_unfinished_guess(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("Let me see.", "en", 0), [])
        self.assertEqual(
            buf.add("Everything arrived okay.", "en", 4),
            [("Let me see.", "en"), ("Everything arrived okay.", "en")]
        )

    def test_safe_in_chunk_restart_cleanup(self):
        repair = self.pipeline.collapse_asr_restarts
        self.assertEqual(
            repair("Yes, I think we can make a We can make an exception this time."),
            "Yes, I think we can make an exception this time."
        )
        self.assertEqual(
            repair("I see. So you So your cash flow problem will be solved."),
            "I see. So you So your cash flow problem will be solved."
        )
        self.assertEqual(
            repair("We can make. We can make an exception."),
            "We can make. We can make an exception."
        )

    def test_repetition_of_short_words_is_preserved(self):
        repair = self.pipeline.collapse_asr_restarts
        self.assertEqual(repair("Yes, yes, thank you."), "Yes, yes, thank you.")
        self.assertEqual(repair("I I think so."), "I I think so.")

    def test_bounded_long_chunk_keeps_unfinished_tail_for_next_audio(self):
        """From user's call: 'cashflow prop' / 'problem will be solved...'."""
        buf = self.pipeline.SentenceBuffer(3)
        buf.max_chars = 70
        self.assertEqual(
            buf.add(
                "This is a new order. The customer pays on delivery. "
                "So your cashflow prop", "en", 0,
            ),
            [("This is a new order. The customer pays on delivery.", "en")],
        )
        self.assertEqual(buf.text, "So your cashflow prop")
        self.assertEqual(
            buf.add("problem will be solved after the new order is delivered.", "en", 4),
            [("So your cashflow problem will be solved after the new order is delivered.", "en")],
        )

    def test_cut_word_join_does_not_rewrite_complete_words(self):
        fix = self.pipeline.SentenceBuffer._carry_incomplete_word
        self.assertEqual(
            fix("So your cashflow prop", "problem will be solved.", "en"),
            ("So your cashflow", "problem will be solved."),
        )
        self.assertEqual(
            fix("The contract ends with prop.", "property is yours.", "en"),
            ("The contract ends with prop.", "property is yours."),
        )
        self.assertEqual(
            fix("The contract will", "will be signed.", "en"),
            ("The contract will", "will be signed."),
        )
        self.assertEqual(
            fix("このプログ", "プログラムを", "ja"),
            ("このプログ", "プログラムを"),
        )

    def test_finite_buffer_still_flushes_when_no_safe_sentence_break(self):
        buf = self.pipeline.SentenceBuffer(3)
        buf.max_chars = 30
        long_phrase = "This is a deliberately long sentence without a full stop"
        self.assertEqual(
            buf.add(long_phrase, "en", 0),
            [(long_phrase, "en")],
        )

    def test_complete_sentence_delivered_without_extra_hold(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("You'd really be helping us if you could extend it to 60 days.",
                    "en", 0),
            [("You'd really be helping us if you could extend it to 60 days.", "en")],
        )

    def test_cash_flow_phrase_is_not_translated_as_positive_cash(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("I know they're usually 30 days, but we're having some cash.",
                    "en", 0), []
        )
        self.assertEqual(
            buf.add("flow problems. Could you extend it to 60 days?", "en", 4),
            [("I know they're usually 30 days, but we're having some cash "
              "flow problems. Could you extend it to 60 days?", "en")]
        )

    def test_cash_flow_at_hard_buffer_limit_preserves_open_phrase(self):
        buf = self.pipeline.SentenceBuffer(3)
        buf.max_chars = 35
        self.assertEqual(
            buf.add("Order looks fine. We're having some cash.", "en", 0),
            [("Order looks fine.", "en")]
        )
        self.assertEqual(buf.text, "We're having some cash.")
        self.assertEqual(
            buf.add("flow problems.", "en", 3),
            [("We're having some cash flow problems.", "en")]
        )

    def test_payment_in_cash_is_not_unnecessarily_held(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("You can pay in cash.", "en", 0),
            [("You can pay in cash.", "en")]
        )

    def test_real_new_utterance_is_not_mistaken_for_cash_flow(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("We have some cash.", "en", 0), [])
        self.assertEqual(
            buf.add("Everything else is ready.", "en", 3),
            [("We have some cash.", "en"), ("Everything else is ready.", "en")]
        )

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

    def test_cash_flow_phrase_joined_across_capture_blocks(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("We have to manage our own cash.", "en", 0), [])
        results = buf.add("flow. New order next.", "en", 2)
        self.assertIn("own cash flow", " ".join(text for text, _ in results))

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

    def test_overlap_retains_context_without_repeating_new_audio(self):
        self.assertAlmostEqual(self.pipeline.QUALITY_OVERLAP_SECONDS, 0.9)
        # Overlap must be context for ASR only; the capture queue still
        # receives exactly one joined audio block, not two recordings.
        source = (Path(__file__).resolve().parent.parent / "streaming_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("block, overlap_tail = add_audio_context(", source)
        self.assertIn("item = (time.monotonic(), block, rate)", source)
        self.assertIn("strip_transcript_overlap(", source)

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


    def test_payment_extension_continues_into_terms(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("I need an extension.", "en", 0), [])
        self.assertEqual(
            buf.add("on the payment terms.", "en", 3),
            [("I need an extension on the payment terms.", "en")],
        )

    def test_payment_extension_does_not_swallow_new_turn(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("I need an extension.", "en", 0), [])
        self.assertEqual(
            buf.add("That's not possible.", "en", 3),
            [("I need an extension.", "en"), ("That's not possible.", "en")],
        )


if __name__ == "__main__":
    unittest.main()

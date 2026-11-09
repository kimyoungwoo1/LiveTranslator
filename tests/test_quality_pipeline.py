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

    def test_ellipsis_releases_completed_prefix_without_extra_latency(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(
            buf.add("Please send the contract. I am not...", "en", 0),
            [("Please send the contract.", "en")],
        )
        self.assertEqual(
            buf.add("sure we can accept it.", "en", 6),
            [("I am not sure we can accept it.", "en")],
        )

    def test_ellipsis_does_not_merge_a_clearly_new_utterance(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(buf.add("We must manage our own...", "en", 0), [])
        self.assertEqual(
            buf.add("I promise this is temporary.", "en", 6),
            [("We must manage our own", "en"),
             ("I promise this is temporary.", "en")],
        )

    def test_meeting_time_joins_recognized_clock_fragments(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(buf.add("We could postpone it to the afternoon to", "en", 0), [])
        self.assertEqual(
            buf.add("1 p.m., for example. What would suit you? Could we make it?", "en", 6),
            [("We could postpone it to the afternoon to 1 p.m., for example. "
              "What would suit you?", "en")],
        )
        self.assertEqual(buf.add("9 o'clock? That would really help me.", "en", 12),
                         [("Could we make it 9 o'clock? That would really help me.", "en")])
        self.assertEqual(buf.add("Could we make it?", "en", 18), [])
        self.assertEqual(buf.add("Yes, definitely.", "en", 24),
                         [("Could we make it?", "en"), ("Yes, definitely.", "en")])

    def test_session_boundary_replay_preserves_continuations_and_new_turns(self):
        cases = (
            ("You'd really be helping us out if you could extend it to",
             "60 days. I'm not sure if I can do that, Andrea.",
             ["You'd really be helping us out if you could extend it to 60 days. "
              "I'm not sure if I can do that, Andrea."]),
            ("Let me see what", "I can do. Yes, I think we can make an exception this time.",
             ["Let me see what I can do. Yes, I think we can make an exception this time."]),
            ("Actually", "I also want to place another new order.",
             ["Actually, I also want to place another new order."]),
            ("That's", "That's great, Junko. I appreciate your help.",
             ["That's great, Junko. I appreciate your help."]),
            ("And", "And I'll email you the new order.",
             ["And I'll email you the new order."]),
            ("We've got regulations and also have to manage our",
             "I promise this won't become the norm.",
             ["We've got regulations and also have to manage our",
              "I promise this won't become the norm."]),
            ("Please wait until I get to", "60 customers are on the line.",
             ["Please wait until I get to", "60 customers are on the line."]),
        )
        for first, following, expected in cases:
            with self.subTest(first=first, following=following):
                buf = self.pipeline.SentenceBuffer(5)
                self.assertEqual(buf.add(first, "en", 0), [])
                self.assertEqual(buf.add(following, "en", 6),
                                 [(text, "en") for text in expected])

    def test_conjunction_joins_capitalized_subject_after_audio_boundary(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(buf.add("I know there are usually 30 days but", "en", 0), [])
        self.assertEqual(
            buf.add("We're having some cash flow problems.", "en", 6),
            [("I know there are usually 30 days but we're having some cash flow problems.", "en")],
        )
        self.assertEqual(buf.add("We have to manage our", "en", 7), [])
        self.assertEqual(buf.add("I promise this won't become the norm.", "en", 13),
                         [("We have to manage our", "en"),
                          ("I promise this won't become the norm.", "en")])

    def test_leading_ellipsis_joins_pending_words_and_numbers(self):
        for first, following, expected in (
            ("This is a little...", "...difficult.", "This is a little difficult."),
            ("I know they're usually...", "…30 days, but cash is tight.",
             "I know they're usually 30 days, but cash is tight."),
            ("I know they're usually", "...30 days.", "I know they're usually 30 days."),
        ):
            with self.subTest(first=first, following=following):
                buf = self.pipeline.SentenceBuffer(5)
                self.assertEqual(buf.add(first, "en", 0), [])
                self.assertEqual(buf.add(following, "en", 6), [(expected, "en")])

    def test_leading_ellipsis_without_pending_audio_stays_separate(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(buf.add("The order arrived.", "en", 0),
                         [("The order arrived.", "en")])
        self.assertEqual(buf.add("...30 days passed.", "en", 6),
                         [("...30 days passed.", "en")])

    def test_forced_cut_replaces_only_verified_clipped_word_restart(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(buf.add("It's for an important custom.", "en", 0,
                                 cut_mid_speech=True), [])
        self.assertEqual(
            buf.add("An important customer and they pay on delivery.", "en", 6),
            [("It's for an important customer and they pay on delivery.", "en")],
        )
        self.assertEqual(buf.add("The order is ready.", "en", 7,
                                 cut_mid_speech=True), [])
        self.assertEqual(buf.add("Thank you.", "en", 13),
                         [("The order is ready.", "en"), ("Thank you.", "en")])

    def test_forced_audio_cut_joins_conjunction_but_releases_prior_sentence(self):
        buf = self.pipeline.SentenceBuffer(5)
        self.assertEqual(
            buf.add("The order arrived. I am calling about the invoice.", "en", 0,
                    cut_mid_speech=True),
            [("The order arrived.", "en")],
        )
        self.assertEqual(
            buf.add("and the payment terms.", "en", 6),
            [("I am calling about the invoice and the payment terms.", "en")],
        )

    def test_long_batch_splits_without_losing_words_or_splitting_titles(self):
        split = self.pipeline._split_long_english_batch
        text = ("We spoke to Dr. Morgan about the delivery. "
                "They will pay in thirty days. The order is ready.")
        units = split(text, limit=30)
        self.assertEqual(" ".join(units), text)
        self.assertEqual(units[0], "We spoke to Dr. Morgan about the delivery.")

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

    def test_trailing_short_questions_are_not_translated_out_of_context(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("Hello? Junko Mori speaking? How?", "en", 0),
            [("Hello? Junko Mori speaking?", "en")],
        )
        self.assertEqual(
            buf.add("How can I help you? Hi, Junko.", "en", 3),
            [("How can I help you? Hi, Junko.", "en")],
        )

    def test_bare_how_restarts_to_complete_question(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("How?", "en", 0), [])
        self.assertEqual(
            buf.add("How can I help you?", "en", 3),
            [("How can I help you?", "en")],
        )

    def test_it_works_both_and_lone_ways_join(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("We appreciate your business. It works both.", "en", 0),
            [("We appreciate your business.", "en")],
        )
        self.assertEqual(
            buf.add("ways.", "en", 3),
            [("It works both ways.", "en")],
        )

    def test_let_me_see_what_i_can_and_lone_do_join(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("Let me see what I can.", "en", 0), [],
        )
        self.assertEqual(
            buf.add("do.", "en", 3),
            [("Let me see what I can do.", "en")],
        )

    def test_happy_to_help_and_lone_you_join(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("Sure. We are happy to help.", "en", 0),
            [("Sure.", "en")],
        )
        self.assertEqual(
            buf.add("you. Great.", "en", 3),
            [("We are happy to help you. Great.", "en")],
        )

    def test_new_speaker_does_not_get_folded_into_open_question(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(buf.add("How?", "en", 0), [])
        self.assertEqual(
            buf.add("Everything arrived safely.", "en", 4),
            [("How?", "en"), ("Everything arrived safely.", "en")],
        )

    def test_uncertain_response_waits_for_following_clause(self):
        buf = self.pipeline.SentenceBuffer(3)
        buf.max_fragments = 3
        self.assertEqual(buf.add("The buyer requests 60 days", "en", 0), [])
        self.assertEqual(buf.add("and the supplier answers", "en", 3), [])
        self.assertEqual(buf.add("I'm not sure", "en", 6), [])
        self.assertEqual(
            buf.add("if I can do that, Andrea.", "en", 9),
            [("The buyer requests 60 days and the supplier answers I'm not sure if I can do that, Andrea.", "en")],
        )

    def test_complete_uncertain_response_remains_immediate(self):
        buf = self.pipeline.SentenceBuffer(3)
        self.assertEqual(
            buf.add("I'm not sure if I can do that, Andrea.", "en", 0),
            [("I'm not sure if I can do that, Andrea.", "en")],
        )

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
        self.assertIn("item = (time.monotonic(), block, rate, cut_mid_speech)", source)
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

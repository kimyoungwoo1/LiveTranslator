"""Test optional Hy-MT2 local-only backend without downloading any model or binary."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load_backend():
    spec = importlib.util.spec_from_file_location("hym_backend_test", ROOT / "streaming_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "numpy": mock.MagicMock(),
        "scipy": mock.MagicMock(),
        "scipy.signal": mock.MagicMock(),
        "pyaudiowpatch": mock.MagicMock(),
    }):
        spec.loader.exec_module(module)
    return module


class HyMT2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_backend()

    def test_model_and_runtime_are_optional(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(self.mod.hym_ready(temp))
            self.assertIsNone(self.mod.find_hym_server(temp))
            self.assertFalse((Path(temp) / "models").exists())

    def test_model_ready_with_local_server_without_external_download(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            exe = folder / "runtime" / "llama-server.exe"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"not an executable")
            model = folder / "models" / self.mod.HYM_FILE
            model.parent.mkdir(parents=True)
            with model.open("wb") as handle:
                handle.truncate(500_000_001)
            self.assertTrue(self.mod.hym_ready(folder))

    def test_xet_metadata_head_supplies_sha_without_lfs_tree(self):
        digest = "a" * 64
        response = mock.MagicMock()
        response.headers = {
            "X-Linked-Etag": '"' + digest + '"',
            "X-Linked-Size": "1130000000",
        }
        opener = mock.MagicMock()
        opener.open.return_value = response
        with mock.patch.object(
            self.mod.urlrequest, "build_opener", return_value=opener
        ), mock.patch.object(self.mod, "_hym_json", side_effect=AssertionError(
            "Xet metadata does not need LFS tree"
        )):
            result = self.mod._hym_model_metadata()
        self.assertEqual(result, (digest, 1130000000))
        self.assertEqual(opener.open.call_args.args[0].get_method(), "HEAD")

    def test_hugging_face_xet_redirect_302_contains_verified_metadata(self):
        # Hugging Face Xet serves metadata on the *redirect* response; an
        # automatic redirect to the CDN would lose these headers.
        from urllib.error import HTTPError
        digest = "c" * 64
        headers = {
            "X-Linked-Etag": '"' + digest + '"',
            "X-Linked-Size": "1130000000",
        }
        redirected = HTTPError(
            self.mod.HYM_MODEL_URL, 302, "Found", headers, None
        )
        opener = mock.MagicMock()
        opener.open.side_effect = redirected
        with mock.patch.object(
            self.mod.urlrequest, "build_opener", return_value=opener
        ), mock.patch.object(self.mod, "_hym_json",
                            side_effect=AssertionError("No tree request")):
            result = self.mod._hym_model_metadata()
        self.assertEqual(result, (digest, 1130000000))

    def test_windows_cpu_asset_sha256_from_official_release(self):
        asset = {
            "name": "llama-b12122-bin-win-cpu-x64.zip",
            "digest": "sha256:" + "b" * 64,
            "browser_download_url":
                "https://github.com/ggml-org/llama.cpp/releases/download/b12122/"
                "llama-b12122-bin-win-cpu-x64.zip",
        }
        with mock.patch.object(
            self.mod, "_hym_json", return_value={"assets": [asset]}
        ):
            url, digest, name = self.mod._hym_runtime_asset()
        self.assertEqual(name, asset["name"])
        self.assertEqual(digest, "b" * 64)
        self.assertEqual(url, asset["browser_download_url"])

    def test_official_sha_pinned_fallback_when_release_api_has_no_digest(self):
        with mock.patch.object(
            self.mod, "_hym_json",
            return_value={"assets": [{
                "name": "llama-b12122-bin-win-cpu-x64.zip",
                "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b12122/llama-b12122-bin-win-cpu-x64.zip",
                "digest": None,
            }]}
        ):
            result = self.mod._hym_runtime_asset()
        self.assertEqual(result, self.mod.HYM_PINNED_WIN_CPU)
        self.assertEqual(len(result[1]), 64)

    def test_official_sha_pinned_fallback_when_api_is_blocked(self):
        with mock.patch.object(
            self.mod, "_hym_json", side_effect=OSError("HTTP 403")
        ):
            result = self.mod._hym_runtime_asset()
        self.assertEqual(result, self.mod.HYM_PINNED_WIN_CPU)

    def test_installer_error_is_visible_and_saved(self):
        src = (ROOT / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('text="설치 오류 확인"', src)
        self.assertIn('install_error.log', src)
        self.assertIn('self.status.set(f"Hy-MT2 설치 실패 · {data[:190]}")', src)

    def test_only_trusted_official_sha256_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.mod._hym_download(
                    "https://github.com/ggml-org/llama.cpp/releases/download/x/test.zip",
                    Path(tmp) / "test.zip", "", 100, lambda _: None
                )
            with self.assertRaisesRegex(RuntimeError, "허용되지 않은"):
                self.mod._hym_download(
                    "https://example.org/malicious.zip",
                    Path(tmp) / "malicious.zip", "a" * 64, 100, lambda _: None
                )

    def test_cpu_server_translates_via_localhost_and_preserves_numbers(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "결제 기한을 60일로 연장해 주세요."}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        with mock.patch.object(
            self.mod.urlrequest, "urlopen", return_value=manager
        ) as urlopen:
            result = backend.translate("Please extend the payment terms to 60 days.", "en")
        self.assertEqual(result, "결제 기한을 60일로 연장해 주세요.")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:14811/v1/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(body["messages"][0]["content"].endswith("60 days."))
        self.assertFalse(body["stream"])

    def test_translation_prompt_preserves_negation_without_past_speech(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "결제 기한을 연장할 수 있을지 모르겠습니다."}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        original = "I'm not sure if I can do that, Andrea."
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            backend.translate(original, "en")
        prompt = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertTrue(prompt.endswith(original))
        self.assertIn("Preserve negation", prompt)
        self.assertIn("Do not guess or add missing speech.", prompt)
        self.assertIn("natural, polite Korean", prompt)
        self.assertNotIn("PREVIOUS", prompt)
        self.assertNotIn("CURRENT (translate this only)", prompt)

    def test_glossary_matches_phrases_and_not_ambiguous_words(self):
        hint = self.mod._hym_glossary_hint
        self.assertIn("주문하다", hint("They PLACED another new order.", "en"))
        self.assertIn("물건을 받을 때 결제하다", hint("They pay  on delivery.", "en"))
        self.assertIn("현금 흐름 문제", hint("We have cash-flow problems.", "en"))
        self.assertEqual(hint("Run these commands in order.", "en"), "")
        self.assertEqual(hint("They repay on delivery.", "en"), "")
        self.assertEqual(hint("They pay on delivery.", "ja"), "")

    def test_business_idiom_guidance_uses_only_current_utterance(self):
        pick = self.mod._hym_current_text_guidance
        current = "And we appreciate your business, Andrea. It works both ways."
        guidance = pick(current, "en")
        self.assertIn("mutual benefit", guidance)
        self.assertNotIn("payment terms", guidance)
        self.assertEqual(pick("This program works both ways.", "en"), "")
        self.assertEqual(pick("この問題を解決できます。", "ja"), "")
        self.assertIn(
            "customer's patronage",
            pick("We appreciate your business, Andrea.", "en")
        )
        self.assertIn("watch for", pick("I'll keep an eye out for it.", "en"))

    def test_business_hint_does_not_include_prior_speech_or_modify_source(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "서로에게 도움이 됩니다."}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        prompts = []

        def record(req, **kwargs):
            prompts.append(json.loads(req.data)["messages"][0]["content"])
            return manager

        prior = "Please extend the payment terms to 60 days."
        present = "We appreciate your business, Andrea. It works both ways."
        with mock.patch.object(self.mod.urlrequest, "urlopen", side_effect=record):
            backend.translate(prior, "en")
            backend.translate(present, "en")
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[1].endswith(present))
        self.assertIn("mutual benefit", prompts[1])
        self.assertNotIn(prior, prompts[1])
        self.assertNotIn("PREVIOUS (context only)", prompts[1])

    def test_never_sends_previous_utterance_after_sequential_translations(self):
        """Regression: v0.6.23 repeated old Korean lines instead of translating CURRENT."""
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "결제 기한을 60일로 연장해 주세요."}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        prompts = []

        def record(req, **kwargs):
            prompts.append(json.loads(req.data)["messages"][0]["content"])
            return manager

        lines = (
            "I'm calling about the invoice and payment terms.",
            "Extend the payment terms to 60 days.",
            "I'm not sure if I can do that, Andrea.",
            "Yes, I think we can make an exception this time.",
            "Sure, no problem. We're happy to help you.",
            "I'll keep an eye out for it.",
        )
        with mock.patch.object(self.mod.urlrequest, "urlopen", side_effect=record):
            for utterance in lines:
                backend.translate(utterance, "en")
        self.assertEqual(len(prompts), len(lines))
        for index, utterance in enumerate(lines):
            with self.subTest(index=index):
                self.assertTrue(prompts[index].endswith(utterance))
                self.assertNotIn("PREVIOUS", prompts[index])
                self.assertNotIn("CURRENT (translate", prompts[index])
                for other_index, other in enumerate(lines):
                    if index != other_index:
                        self.assertNotIn(other, prompts[index])
        self.assertFalse(hasattr(backend, "_source_context"))

    def test_switching_languages_does_not_share_older_text(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "ありがとうございます"}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            backend.translate("Thank you for your business.", "en")
            backend.translate("ありがとうございます。", "ja")
        japanese_prompt = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertNotIn("Thank you for your business.", japanese_prompt)
        self.assertTrue(japanese_prompt.endswith("ありがとうございます。"))
        backend.stop()

    def test_failed_utterance_is_not_passed_to_next_request(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = b'{"choices":[{"message":{"content":""}}]}'
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "비어"):
                backend.translate("Unrecognized prior phrase.", "en")
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "예외를 허용할 수 있습니다."}}]
        }).encode("utf-8")
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            backend.translate("We can make an exception this time.", "en")
        next_prompt = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertNotIn("Unrecognized prior phrase.", next_prompt)
        self.assertTrue(next_prompt.endswith("We can make an exception this time."))

    def test_summary_uses_local_backend_and_only_current_korean_passage(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "핵심 내용: 결제 기한을 60일로 연장."}}]
        }, ensure_ascii=False).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            result = backend.summarize_korean("결제 기한을 60일로 연장하기로 했습니다.")
        self.assertIn("60일", result)
        prompt = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertIn("요약", prompt)
        self.assertIn("없는 내용을 추측하거나 만들어 내지 마세요", prompt)
        self.assertIn("결제 기한을 60일로 연장하기로 했습니다.", prompt)
        self.assertNotIn("PREVIOUS", prompt)

    def test_summary_prompt_preserves_proposal_vs_confirmed_and_order_details(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "결정/협의 상태: 60일 연장 잠정 동의"}}]
        }, ensure_ascii=False).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            backend.summarize_korean(
                "기존 30일 결제 기한을 60일로 연장 요청. 이번에는 예외를 허용할 수 있을 것 같아요.",
                evidence="신규 주문은 지난번과 같은 규모. 신규 주문서를 이메일로 보내겠습니다.",
            )
        prompt = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertIn("잠정 수락", prompt)
        self.assertIn("최종 승인", prompt)
        self.assertIn("신규 주문", prompt)
        self.assertIn("후속 조치", prompt)
        self.assertIn("지난번과 같은 규모", prompt)
        self.assertIn("30일", prompt)
        self.assertNotIn("PREVIOUS", prompt)

    def test_literal_idiom_fix_does_not_touch_bidirectional_software(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        self.assertEqual(
            fix("This program works both ways. Thanks!",
                "양방향으로 작동해요.", "en"),
            "양방향으로 작동해요.",
        )
        self.assertEqual(
            fix("And we appreciate your business, Andrea.",
                "그리고 당신의 고객님의 지원에 감사드립니다, 안드레아.", "en"),
            "그리고 거래해 주셔서 감사합니다, 안드레아.",
        )
        self.assertEqual(
            fix("Thanks. I'll keep an eye out for it.",
                "고마워요. 그걸 기대하며 주의를 기울이겠습니다.", "en"),
            "고마워요. 소식 기다릴게요.",
        )

    def test_invoice_and_payment_term_guidance_is_source_gated(self):
        hint = self.mod._hym_current_text_guidance
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        self.assertIn(
            "청구서",
            hint("I'm calling about the invoice and the payment terms.", "en"),
        )
        self.assertIn(
            "payment deadline",
            hint("I need an extension on the payment terms.", "en"),
        )
        self.assertEqual(
            fix("The invoice and payment terms arrived.",
                "영수증과 결제 조건이 도착했습니다.", "en"),
            "청구서와 결제 조건이 도착했습니다.",
        )
        # A sentence mentioning both should never replace the actual receipt.
        self.assertEqual(
            fix("The invoice and the receipt arrived.",
                "청구서와 영수증이 도착했습니다.", "en"),
            "청구서와 영수증이 도착했습니다.",
        )
        self.assertEqual(
            fix("This is your receipt.", "영수증입니다.", "en"),
            "영수증입니다.",
        )
        self.assertEqual(
            fix("The invoice arrived.", "영수증입니다.", "ja"),
            "영수증입니다.",
        )

    def test_spoken_business_idiom_variants_from_live_capture(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        self.assertEqual(
            fix("It works both ways.", "양방향으로 잘 작동해요.", "en"),
            "서로에게 도움이 되는 일이에요.",
        )
        self.assertEqual(
            fix("And I'll email you the new order. Thanks. I'll keep an eye out for it.",
                "새 주문 내용을 이메일로 보낼게요. 고마워요. 그걸 기대하며 지켜볼게요.", "en"),
            "새 주문 내용을 이메일로 보낼게요. 고마워요. 메일이 오면 확인할게요.",
        )
        self.assertEqual(
            fix("This machine works both ways.", "양방향으로 잘 작동해요.", "en"),
            "양방향으로 잘 작동해요.",
        )

    def test_business_request_roles_and_proper_name_are_preserved(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        hint = self.mod._hym_current_text_guidance
        self.assertEqual(
            fix("I need a favor. A favor?",
                "도와줄 일이 있어요. 도와주시는 거예요?", "en"),
            "부탁 하나 드리고 싶어요. 부탁이라고요?",
        )
        self.assertEqual(
            fix("From Red Band.", "빨간 밴드에서 전화했어요.", "en"),
            "레드 밴드에서 전화했어요.",
        )
        self.assertEqual(
            fix("The red band is broken.", "빨간 밴드가 고장 났어요.", "en"),
            "빨간 밴드가 고장 났어요.",
        )
        self.assertIn("request for help", hint("I need a favor.", "en"))

    def test_long_business_dialogue_translation_repairs_only_source_supported_meanings(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        source = ("That's great, Junko. We appreciate your business, Andrea. "
                  "It works both ways. Thanks again.")
        korean = ("정말 좋네요, 준코. 우리도 당신의 비즈니스에 감사하죠, 안드레아. "
                  "이건 양방향으로 작동하는 관계예요. 다시 한번 감사합니다.")
        output = fix(source, korean, "en")
        self.assertIn("거래해 주셔서 감사합니다", output)
        self.assertIn("서로에게 도움이 되는 관계예요", output)
        self.assertNotIn("양방향으로 작동", output)

    def test_quoted_favor_reply_does_not_duplicate_request(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        self.assertEqual(
            fix("I need a favor. A favor?",
                "부탁 하나 드리고 싶어요. 부탁 하나 드리고 싶어요?", "en"),
            "부탁 하나 드리고 싶어요. 부탁이라고요?",
        )
        self.assertEqual(
            fix("The machine works both ways.",
                "이건 양방향으로 작동하는 관계예요.", "en"),
            "이건 양방향으로 작동하는 관계예요.",
        )

    def test_unsupported_language_does_not_send_requests(self):
        backend = self.mod.HyMT2Backend("unused")
        with self.assertRaisesRegex(RuntimeError, "영어 또는 일본어"):
            backend.translate("Bonjour", "fr")

    def test_server_shutdown_terminates_owned_process(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        backend.log = mock.MagicMock()
        backend.stop()
        self.assertIsNone(backend.process)
        self.assertIsNone(backend.port)

    def test_live_translator_option_and_install_are_explicit(self):
        text = (ROOT / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('value="hymt"', text)
        self.assertIn('text="Hy-MT2 설치"', text)
        self.assertIn('command=self.install_hymt', text)
        self.assertIn('messagebox.askyesno(', text)
        self.assertIn('HyMT2Backend(', text)
        self.assertIn('hy_backend.stop()', text)
        self.assertIn('quality_mode=(engine in ("argos", "hymt"))', text)
        self.assertIn('if engine == "google":', text)


    def test_conservative_korean_idiom_repair(self):
        fix = self.mod._repair_clear_korean_idiom_mistranslations
        self.assertEqual(
            fix("We appreciate your business, Andrea.",
                "여러분의 고객이 되어주셔서 감사합니다.", "en"),
            "거래해 주셔서 감사합니다.",
        )
        self.assertEqual(
            fix("It works both ways. Thanks again, Junko.",
                "양방으로 작동해요. 다시 한번 고마워요.", "en"),
            "서로에게 도움이 되는 일이에요. 다시 한번 고마워요.",
        )
        self.assertEqual(
            fix("I'll email you the new order. I'll keep an eye out for it.",
                "새 주문을 이메일로 보낼게요. 그걸 잘 지켜보겠습니다.", "en"),
            "새 주문을 이메일로 보낼게요. 메일이 오면 확인할게요.",
        )
        self.assertEqual(
            fix("It works both ways.", "양방향으로 작동해요.", "en"),
            "서로에게 도움이 되는 일이에요.",
        )
        self.assertEqual(
            fix("This program works both ways. Thanks!",
                "양방으로 작동해요.", "en"),
            "양방으로 작동해요.",
        )
        self.assertEqual(
            fix("We appreciate your business.", "거래해 주셔서 고맙습니다.", "en"),
            "거래해 주셔서 고맙습니다.",
        )
        self.assertEqual(
            fix("We appreciate your business.", "여러분의 고객이 되어주셔서 감사합니다.", "ja"),
            "여러분의 고객이 되어주셔서 감사합니다.",
        )


if __name__ == "__main__":
    unittest.main()

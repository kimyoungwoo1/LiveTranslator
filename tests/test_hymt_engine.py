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

    def test_context_is_bounded_and_previous_dialogue_not_translated(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = None
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "상호 도움이 되는 관계입니다."}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        requests = []

        def record(req, **kwargs):
            requests.append(json.loads(req.data)["messages"][0]["content"])
            return manager

        with mock.patch.object(self.mod.urlrequest, "urlopen", side_effect=record):
            backend.translate("And we appreciate your business, Andrea.", "en")
            backend.translate("It works both ways.", "en")
            backend.translate("I'll keep an eye out for it.", "en")

        self.assertNotIn("PREVIOUS (context only):", requests[0])
        self.assertIn("CURRENT (translate this only):", requests[0])
        self.assertIn("PREVIOUS (context only):", requests[1])
        self.assertIn("And we appreciate your business, Andrea.", requests[1])
        self.assertTrue(requests[1].endswith("CURRENT (translate this only):\nIt works both ways."))
        self.assertTrue(requests[2].endswith("CURRENT (translate this only):\nI'll keep an eye out for it."))
        self.assertIn("never translate or repeat PREVIOUS", requests[2])
        self.assertEqual(len(backend._source_context), 2)

    def test_context_is_not_carried_across_languages_or_sessions(self):
        backend = self.mod.HyMT2Backend("unused")
        backend.port = 14811
        backend.process = mock.MagicMock()
        backend.process.poll.return_value = 0
        # Simulate a finished Japanese translation in the same engine object.
        backend._source_context = ["Thank you for your business."]
        backend._source_context_lang = "en"
        response = mock.MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "どうもありがとうございます"}}]
        }).encode("utf-8")
        manager = mock.MagicMock()
        manager.__enter__.return_value = response
        backend.process.poll.return_value = None
        with mock.patch.object(self.mod.urlrequest, "urlopen", return_value=manager) as opened:
            backend.translate("ありがとうございます。", "ja")
        sent = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertNotIn("Thank you for your business.", sent)
        self.assertEqual(backend._source_context_lang, "ja")
        self.assertEqual(backend._source_context, ["ありがとうございます。"])
        backend.stop()
        self.assertEqual(backend._source_context, [])
        self.assertIsNone(backend._source_context_lang)

    def test_translation_failure_does_not_pollute_following_context(self):
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
                backend.translate("Not recognized.", "en")
        self.assertEqual(backend._source_context, [])

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


if __name__ == "__main__":
    unittest.main()

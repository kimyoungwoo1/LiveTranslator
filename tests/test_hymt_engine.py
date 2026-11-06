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

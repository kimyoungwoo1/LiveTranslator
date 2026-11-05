"""Check Google translation option using mocks; no live API request or billable usage."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest import mock
from urllib.error import HTTPError

# Mock heavyweight audio/native dependencies so the update feature can be
# validated using standard Python without reinstalling 800MB of libraries.
with mock.patch.dict(sys.modules, {
    "numpy": mock.MagicMock(),
    "scipy": mock.MagicMock(),
    "scipy.signal": mock.MagicMock(),
    "pyaudiowpatch": mock.MagicMock(),
}):
    import app_core


class GoogleOptionTests(unittest.TestCase):
    def test_google_translation_uses_official_https_api_and_key_header(self):
        content = json.dumps({
            "data": {"translations": [{"translatedText": "안녕하세요 &amp; 반갑습니다"}]}
        }).encode("utf-8")
        response = mock.MagicMock()
        response.read.return_value = content
        handler = mock.MagicMock()
        handler.__enter__.return_value = response
        with mock.patch.object(app_core.urlrequest, "urlopen", return_value=handler) as mock_url:
            answer = app_core.google_translate_text("Hello & welcome", "en", "not-a-real-key")
        self.assertEqual(answer, "안녕하세요 & 반갑습니다")
        args, kwargs = mock_url.call_args
        request = args[0]
        self.assertEqual(request.full_url, app_core.GOOGLE_TRANSLATE_URL)
        self.assertTrue(request.full_url.startswith("https://"))
        self.assertNotIn("not-a-real-key", request.full_url)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("X-goog-api-key"), "not-a-real-key")
        self.assertEqual(kwargs["timeout"], 8)
        payload = json.loads(request.data)
        self.assertEqual(payload["q"], "Hello & welcome")
        self.assertEqual(payload["source"], "en")
        self.assertEqual(payload["target"], "ko")
        self.assertEqual(payload["format"], "text")

    def test_google_translation_http_429_explains_quota(self):
        response = HTTPError(
            app_core.GOOGLE_TRANSLATE_URL, 429, "rate limited", {}, None
        )
        with mock.patch.object(app_core.urlrequest, "urlopen", side_effect=response):
            with self.assertRaisesRegex(RuntimeError, "요청 한도"):
                app_core.google_translate_text("Hello", "en", "not-a-real-key")

    def test_secret_is_not_stored_in_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(app_core, "APP_DIR", Path(tmp)), mock.patch.object(
                app_core, "GOOGLE_KEY_FILE", Path(tmp) / "google_api_key.dpapi"
            ):
                if os.name == "nt":
                    app_core.store_google_key("fake-api-key-used-only-for-unit-test")
                    self.assertEqual(app_core.read_google_key(), "fake-api-key-used-only-for-unit-test")
                    self.assertNotIn(b"fake-api-key", app_core.GOOGLE_KEY_FILE.read_bytes())
                else:
                    with self.assertRaises(RuntimeError):
                        app_core.store_google_key("fake-api-key-used-only-for-unit-test")


if __name__ == "__main__":
    unittest.main()

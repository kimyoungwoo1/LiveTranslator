"""Mocked device-selection tests. No physical speaker or microphone required."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


def import_audio_module():
    path = Path(__file__).resolve().parent.parent / "streaming_pipeline.py"
    spec = importlib.util.spec_from_file_location("streaming_pipeline_device_test", path)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "numpy": mock.MagicMock(),
        "scipy": mock.MagicMock(),
        "scipy.signal": mock.MagicMock(),
        "pyaudiowpatch": mock.MagicMock(),
    }):
        spec.loader.exec_module(mod)
    return mod


class AudioDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_audio_module()

    def test_resolves_same_loopback_name_when_index_changes(self):
        old = ("🔊 출력 | speaker [Loopback]", 34, True, "speaker [Loopback]")
        current = {"name": "speaker [Loopback]", "index": 42,
                   "maxInputChannels": 8, "defaultSampleRate": 48000}
        pa = mock.MagicMock()
        pa.get_loopback_device_info_generator.return_value = iter([current])
        self.assertEqual(self.module.resolve_capture_device(pa, old), current)

    def test_disconnected_output_does_not_capture_some_other_device(self):
        selected = ("🔊 출력 | AirPods [Loopback]", 12, True, "AirPods [Loopback]")
        available = {"name": "7.1 Surround [Loopback]", "index": 12,
                     "maxInputChannels": 8, "defaultSampleRate": 48000}
        pa = mock.MagicMock()
        pa.get_loopback_device_info_generator.return_value = iter([available])
        with self.assertRaisesRegex(RuntimeError, "연결 해제"):
            self.module.resolve_capture_device(pa, selected)

    def test_use_native_8_channel_wasapi_first(self):
        info = {"name": "7.1 Surround [Loopback]", "index": 42,
                "maxInputChannels": 8, "defaultSampleRate": 48000}
        pa = mock.MagicMock()
        opened = self.module.open_capture_stream(pa, info)
        self.assertEqual(opened[1:4], (48000, 8, 4800))
        self.assertEqual(pa.open.call_args.kwargs["channels"], 8)
        self.assertEqual(pa.open.call_args.kwargs["input_device_index"], 42)
        self.assertEqual(pa.open.call_args.kwargs["format"], self.module.pyaudio.paInt16)

    def test_retry_stereo_when_native_channels_are_unavailable(self):
        info = {"name": "speaker [Loopback]", "index": 4,
                "maxInputChannels": 8, "defaultSampleRate": 48000}
        pa = mock.MagicMock()
        pa.open.side_effect = [OSError(-9998, "Invalid number of channels"),
                               OSError(-9998, "Invalid number of channels"),
                               mock.sentinel.stream]
        stream, rate, channels, frames, _ = self.module.open_capture_stream(pa, info)
        self.assertIs(stream, mock.sentinel.stream)
        self.assertEqual(channels, 2)
        self.assertEqual(pa.open.call_count, 3)

    def test_device_open_error_is_readable(self):
        info = {"name": "7.1 Surround [Loopback]", "index": 4,
                "maxInputChannels": 8, "defaultSampleRate": 48000}
        pa = mock.MagicMock()
        pa.open.side_effect = OSError(-9996, "Invalid device")
        with self.assertRaisesRegex(RuntimeError, "Windows의 실제 출력 장치"):
            self.module.open_capture_stream(pa, info)


if __name__ == "__main__":
    unittest.main()

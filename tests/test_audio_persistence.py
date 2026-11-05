"""Audio selection survives restart; playback is auto-discovered only if no preference."""
import sys
import unittest
from unittest import mock

with mock.patch.dict(sys.modules, {
    "numpy": mock.MagicMock(),
    "scipy": mock.MagicMock(),
    "scipy.signal": mock.MagicMock(),
    "pyaudiowpatch": mock.MagicMock(),
}):
    import app_core


class FakeCombo:
    def __init__(self, index=0):
        self.index = index
        self.values = []

    def current(self, new_index=None):
        if new_index is not None:
            self.index = new_index
        return self.index

    def __setitem__(self, item, value):
        if item == "values":
            self.values = list(value)


class AudioSelectionTests(unittest.TestCase):
    def test_saved_razer_is_found_after_portaudio_reindex(self):
        devices = [
            ("🔊 기본 출력 | Virtual 7.1", 12, True, "Virtual 7.1"),
            ("🔊 출력 | Razer Barracuda X", 73, True, "Razer Barracuda X"),
        ]
        self.assertEqual(app_core.find_saved_device_index(
            devices, {"name": "Razer Barracuda X", "loopback": True}), 1)
        self.assertIsNone(app_core.find_saved_device_index(
            devices, {"name": "Disconnected", "loopback": True}))

    def test_refresh_restores_previously_saved_razer_output(self):
        app = app_core.App.__new__(app_core.App)
        app.devices = []
        app.device_combo = FakeCombo()
        app.status = mock.MagicMock()
        app.settings = {"audio_device": {"name": "Razer Barracuda X", "loopback": True}}
        virtual = {"name": "Virtual 7.1", "index": 12, "maxInputChannels": 8}
        razer = {"name": "Razer Barracuda X", "index": 73, "maxInputChannels": 2}
        pa = mock.MagicMock()
        pa.get_loopback_device_info_generator.return_value = iter([virtual, razer])
        pa.get_default_wasapi_loopback.return_value = virtual
        pa.get_device_count.return_value = 0
        with mock.patch.object(app_core.pyaudio, "PyAudio", return_value=pa):
            app_core.App.refresh(app)
        self.assertEqual(app.device_combo.current(), 1)
        self.assertEqual(app.devices[1][3], "Razer Barracuda X")
        self.assertFalse(app._needs_auto_source)

    def test_confirmed_choice_is_saved_on_selection(self):
        app = app_core.App.__new__(app_core.App)
        app.devices = [
            ("🔊 기본 출력 | 7.1", 5, True, "7.1"),
            ("🔊 출력 | Razer Barracuda X", 47, True, "Razer Barracuda X"),
        ]
        app.device_combo = FakeCombo(1)
        app.settings = {}
        app._needs_auto_source = True
        app.persist = mock.MagicMock()
        app_core.App.remember_selected_device(app)
        self.assertEqual(app.settings["audio_device"], {
            "name": "Razer Barracuda X", "loopback": True
        })
        self.assertFalse(app._needs_auto_source)
        app.persist.assert_called_once()

    def test_auto_finder_chooses_only_audible_output_and_autostarts(self):
        app = app_core.App.__new__(app_core.App)
        virtual = ("🔊 기본 출력 | 7.1", 5, True, "7.1")
        razer = ("🔊 출력 | Razer Barracuda X", 47, True, "Razer Barracuda X")
        app.devices = [virtual, razer]
        app.device_combo = FakeCombo(0)
        app.status = mock.MagicMock()
        app.root = mock.MagicMock()
        app.remember_selected_device = mock.MagicMock()
        app_core.App.use_detected_audio_and_start(app, [
            {"device": virtual, "rms": 0.00001, "name": "7.1"},
            {"device": razer, "rms": 0.065, "name": "Razer Barracuda X"},
        ])
        self.assertEqual(app.device_combo.current(), 1)
        app.remember_selected_device.assert_called_once()
        self.assertEqual(app.root.after.call_args.args[0], 50)

    def test_initial_start_initiates_finder_without_translation_worker(self):
        app = app_core.App.__new__(app_core.App)
        app.running = False
        app.devices = [("🔊 기본 출력 | 7.1", 1, True, "7.1")]
        app.device_combo = FakeCombo(0)
        app._audio_scan_busy = False
        app._needs_auto_source = True
        app.scan_audio = mock.MagicMock()
        app_core.App.start(app)
        app.scan_audio.assert_called_once_with(auto_start=True)

    def test_saved_output_does_not_scan_every_start(self):
        app = app_core.App.__new__(app_core.App)
        app.running = False
        app.devices = [("🔊 출력 | Razer", 7, True, "Razer")]
        app.device_combo = FakeCombo(0)
        app._audio_scan_busy = False
        app._needs_auto_source = False
        app.scan_audio = mock.MagicMock()
        app.engine = mock.MagicMock()
        app.engine.get.return_value = "argos"
        app.persist = mock.MagicMock()
        app.stop_signal = None
        app.audio_meter = mock.MagicMock()
        app.audio_state = mock.MagicMock()
        app.start_btn = mock.MagicMock()
        app.stop_btn = mock.MagicMock()
        app.status = mock.MagicMock()
        with mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        app.scan_audio.assert_not_called()
        app.persist.assert_called_once()
        thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()

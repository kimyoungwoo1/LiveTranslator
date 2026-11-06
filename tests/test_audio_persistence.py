"""Audio selection persists and first-start capture never requires sound detection."""
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
        self.assertIn("Razer Barracuda X", app.status.set.call_args.args[0])

    def test_confirmed_choice_is_saved_on_selection(self):
        app = app_core.App.__new__(app_core.App)
        app.devices = [
            ("🔊 기본 출력 | 7.1", 5, True, "7.1"),
            ("🔊 출력 | Razer Barracuda X", 47, True, "Razer Barracuda X"),
        ]
        app.device_combo = FakeCombo(1)
        app.settings = {}
        app.persist = mock.MagicMock()
        app_core.App.remember_selected_device(app)
        self.assertEqual(app.settings["audio_device"], {
            "name": "Razer Barracuda X", "loopback": True
        })
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

    def ready_app(self, devices, saved=None):
        app = app_core.App.__new__(app_core.App)
        app.running = False
        app.devices = devices
        app.device_combo = FakeCombo(0)
        app._audio_scan_busy = False
        app.scan_audio = mock.MagicMock()
        app.settings = {"audio_device": saved} if saved else {}
        app.persist = mock.MagicMock()
        app.refresh = mock.MagicMock()
        app.engine = mock.MagicMock()
        app.engine.get.return_value = "argos"
        app.language = mock.MagicMock()
        app.language.get.return_value = "en"
        app.model = mock.MagicMock()
        app.model.get.return_value = "base"
        app.seconds = mock.MagicMock()
        app.seconds.get.return_value = "3"
        app.audio_meter = mock.MagicMock()
        app.audio_state = mock.MagicMock()
        app.start_btn = mock.MagicMock()
        app.stop_btn = mock.MagicMock()
        app.status = mock.MagicMock()
        app.set_live_indicator = mock.MagicMock()
        return app

    def test_first_start_uses_default_output_without_sound_probe(self):
        default = ("🔊 기본 출력 | Speakers (Realtek)", 12, True,
                   "Speakers (Realtek)")
        app = self.ready_app([default])
        with mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        app.scan_audio.assert_not_called()
        app.refresh.assert_not_called()
        thread.assert_called_once()
        args = thread.call_args.kwargs["args"]
        self.assertEqual(args[0], default)
        self.assertEqual(app.settings["audio_device"],
                         {"name": "Speakers (Realtek)", "loopback": True})
        app.persist.assert_called_once()
        self.assertTrue(app.running)

    def test_first_start_does_not_switch_to_other_output(self):
        default = ("🔊 기본 출력 | Speakers", 12, True, "Speakers")
        other = ("🔊 출력 | Headset", 27, True, "Headset")
        app = self.ready_app([default, other])
        app.device_combo.current(1)
        with mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        thread.assert_called_once()
        self.assertEqual(thread.call_args.kwargs["args"][0], other)
        self.assertEqual(app.settings["audio_device"]["name"], "Headset")
        app.scan_audio.assert_not_called()

    def test_saved_output_starts_without_probe_or_refresh(self):
        source = ("🔊 출력 | Razer", 7, True, "Razer")
        app = self.ready_app(
            [source], saved={"name": "Razer", "loopback": True}
        )
        with mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        app.scan_audio.assert_not_called()
        app.refresh.assert_not_called()
        app.persist.assert_called_once()
        thread.assert_called_once()

    def test_first_start_recovers_missing_enumeration_by_refresh(self):
        source = ("🔊 기본 출력 | Speaker", 8, True, "Speaker")
        app = self.ready_app([])
        app.device_combo = FakeCombo(-1)

        def reload_devices():
            app.devices = [source]
            app.device_combo.current(0)

        app.refresh.side_effect = reload_devices
        with mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        app.refresh.assert_called_once()
        thread.assert_called_once()
        self.assertEqual(thread.call_args.kwargs["args"][0], source)

    def test_no_devices_after_retry_shows_error_but_no_probe(self):
        app = self.ready_app([])
        app.device_combo = FakeCombo(-1)
        with mock.patch.object(app_core.messagebox, "showerror") as alert, \
             mock.patch.object(app_core.threading, "Thread") as thread:
            app_core.App.start(app)
        app.refresh.assert_called_once()
        alert.assert_called_once()
        thread.assert_not_called()
        app.scan_audio.assert_not_called()


    def test_mode_filters_audio_devices_and_remembers_each_kind(self):
        app = app_core.App.__new__(app_core.App)
        speaker = ("🔊 출력 | Speaker", 5, True, "Speaker")
        mic = ("🎤 입력 | Microphone", 6, False, "Microphone")
        app.all_devices = [speaker, mic]
        app.devices = []
        app.device_combo = FakeCombo(-1)
        app.device_combo.set = mock.MagicMock()
        app.mode = mock.MagicMock()
        app.mode.get.return_value = "input"
        app.settings = {"audio_devices_by_mode": {
            "input": {"name": "Microphone", "loopback": False},
            "output": {"name": "Speaker", "loopback": True},
        }}
        app.status = mock.MagicMock()
        app.persist = mock.MagicMock()
        app_core.App._show_mode_devices(app)
        self.assertEqual(app.devices, [mic])
        self.assertEqual(app.device_combo.current(), 0)
        app_core.App.remember_selected_device(app)
        self.assertEqual(app.settings["audio_device"],
                         {"name": "Microphone", "loopback": False})
        self.assertEqual(app.settings["audio_devices_by_mode"]["output"]["name"],
                         "Speaker")
        app.mode.get.return_value = "output"
        app_core.App._show_mode_devices(app)
        self.assertEqual(app.devices, [speaker])

    def test_switching_mode_during_translation_does_not_change_source(self):
        app = app_core.App.__new__(app_core.App)
        app.running = True
        app._audio_scan_busy = False
        app.devices = [("🔊 출력 | Speaker", 5, True, "Speaker")]
        app.device_combo = FakeCombo()
        app.mode = mock.MagicMock()
        app._update_mode_ui = mock.MagicMock()
        app._show_mode_devices = mock.MagicMock()
        app.persist = mock.MagicMock()
        with mock.patch.object(app_core.messagebox, "showinfo"):
            app_core.App.switch_audio_mode(app)
        app.mode.set.assert_called_once_with("output")
        app._show_mode_devices.assert_not_called()
        app.persist.assert_not_called()


if __name__ == "__main__":
    unittest.main()

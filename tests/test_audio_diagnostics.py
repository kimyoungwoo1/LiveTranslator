"""Regression checks for the v0.6.6 live audio diagnostic patch."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent


class AudioDiagnosticsTests(unittest.TestCase):
    def test_audio_level_callback_is_emitted_and_received(self):
        pipe = (ROOT / "streaming_pipeline.py").read_text(encoding="utf-8")
        ui = (ROOT / "app_core.py").read_text(encoding="utf-8")
        ast.parse(pipe)
        ast.parse(ui)
        self.assertIn("emit_audio_level=None", pipe)
        self.assertIn("emit_audio_level(rms, meter_peak)", pipe)
        self.assertIn('("audio_level", (rms, peak))', ui)
        self.assertIn('elif kind == "audio_level":', ui)
        self.assertIn('textvariable=self.audio_state', ui)
        self.assertIn("오디오 데이터 수신 없음", ui)

    def test_quiet_audio_and_surround_phase_guard(self):
        pipe = (ROOT / "streaming_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("rms < 0.0005", pipe)
        self.assertIn("loudest * 0.04", pipe)
        self.assertIn("return multi[:, int(np.argmax(power))]", pipe)
        self.assertIn("음성 인식 결과가 없습니다", pipe)

    def test_engine_selection_retained(self):
        ui = (ROOT / "app_core.py").read_text(encoding="utf-8")
        self.assertIn('quality_mode=(engine in ("argos", "hymt"))', ui)
        self.assertIn('if engine == "google":', ui)
        self.assertIn('self.engine.get()', ui)


if __name__ == "__main__":
    unittest.main()

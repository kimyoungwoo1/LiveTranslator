"""Resolve session quality settings without downloading models or capturing audio."""
import unittest
from unittest import mock
from test_transcript_files import load_app


class RecognitionProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_type = load_app().App

    def resolve(self, model='auto', seconds='3', profile='품질 우선'):
        app = self.app_type.__new__(self.app_type)
        app.model = mock.Mock(); app.model.get.return_value = model
        app.seconds = mock.Mock(); app.seconds.get.return_value = seconds
        app.asr_profile = mock.Mock(); app.asr_profile.get.return_value = profile
        return app._recognition_options()

    def test_quality_uses_small_with_longer_context(self):
        self.assertEqual(self.resolve(), ('small', 5))
        self.assertEqual(self.resolve(seconds='8'), ('small', 8))

    def test_realtime_preserves_previous_auto_behavior(self):
        self.assertEqual(self.resolve(profile='실시간 우선'), ('base', 3))

    def test_explicit_model_overrides_are_respected(self):
        for model in ('base', 'small', 'tiny'):
            with self.subTest(model=model):
                self.assertEqual(self.resolve(model=model), (model, 5))


if __name__ == '__main__':
    unittest.main()

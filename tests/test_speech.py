import numpy as np
import pytest
import soundfile as sf

from storycut import speech


@pytest.fixture(autouse=True)
def clear_speech_cache():
    speech.synthesize.cache_clear()
    yield
    speech.synthesize.cache_clear()


def test_measurement_and_render_share_speech_but_voice_and_edits_invalidate_it(tmp_path, monkeypatch):
    calls = []
    class Voice:
        def create(self, text, **kwargs):
            calls.append((text, kwargs["voice"]))
            return np.ones(12000, dtype=np.float32) * 0.1, 24000
    monkeypatch.setattr(speech, "load_voice", Voice)
    texts = ["First scene.", "Second scene."]
    preview = tmp_path / "preview.wav"
    durations = speech.measure_narration(texts, "Heart · American", preview)
    assert durations == [0.62, 0.62]
    assert sf.info(preview).duration == pytest.approx(1.24)
    duration, captions = speech.narrate(texts[0], "Heart · American", tmp_path / "scene.wav")
    assert duration == 0.62
    assert captions == [(0.0, 0.5, texts[0])]
    assert len(calls) == 2
    speech.measure_narration(texts[:1], "Emma · British")
    speech.measure_narration(["Edited scene."], "Heart · American")
    assert len(calls) == 4

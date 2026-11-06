import json
import zipfile

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from storycut import render
from storycut.storyboard import Storyboard


@pytest.mark.parametrize("aspect,dimensions", list(render.FORMATS.items()))
def test_export_contains_playable_video_and_aligned_captions(tmp_path, monkeypatch, aspect, dimensions):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    def speech(text, voice, destination):
        sf.write(destination, np.zeros(12_000, dtype=np.float32), 24000)
        return 0.5, [(0.0, 0.5, text)]
    monkeypatch.setattr(render, "narrate", speech)
    image = tmp_path / "input.png"
    Image.new("RGB", (300, 300), "#74a164").save(image)
    board = Storyboard(title="Test video", scenes=[
        {"heading": "First", "narration": "One short sentence.", "visual_query": "trees"},
        {"heading": "Second", "narration": "Another short sentence.", "visual_query": "trees"},
    ])
    video, archive, _, stats = render.render_storyboard(board, "test", aspect, [str(image)] * 2)
    reader = imageio_ffmpeg.read_frames(video)
    metadata = next(reader)
    assert metadata["size"] == dimensions
    assert abs(metadata["duration"] - 1.0) < 0.1
    assert len(list(reader)) == 24
    assert stats["scene_seconds"] == [0.5, 0.5]
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"storycut.mp4", "captions.srt", "storyboard.json", "credits.txt"}
        assert "00:00:00,500 --> 00:00:01,000" in bundle.read("captions.srt").decode()
        assert json.loads(bundle.read("storyboard.json"))["title"] == "Test video"


def test_failed_render_removes_partial_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    def fail(*_):
        raise ValueError("Image unavailable")
    monkeypatch.setattr(render, "commons_visual", fail)
    from storycut.storyboard import EXAMPLE
    with pytest.raises(ValueError, match="Image unavailable"):
        render.render_storyboard(EXAMPLE, "test", "Portrait · 9:16", None)
    assert not list(render.OUTPUTS.iterdir())


def test_subtitle_timestamp_carries_across_minutes():
    assert render.srt_time(59.9996) == "00:01:00,000"

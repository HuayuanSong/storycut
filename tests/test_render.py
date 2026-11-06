import json
import zipfile

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from storycut import render
from storycut.storyboard import Storyboard


@pytest.mark.parametrize("ai_motion", [False, True])
@pytest.mark.parametrize("aspect,dimensions", list(render.FORMATS.items()))
def test_export_contains_playable_video_and_aligned_captions(tmp_path, monkeypatch, aspect, dimensions, ai_motion):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    def speech(text, voice, destination):
        sf.write(destination, np.zeros(12_000, dtype=np.float32), 24000)
        return 0.5, [(0.0, 0.5, text)]
    monkeypatch.setattr(render, "narrate", speech)
    def motion(source, scene, size, destination, seed):
        assert ai_motion, "Photo mode must never request video generation"
        # A real moving MP4, deliberately shorter than narration to exercise retiming.
        writer = imageio_ffmpeg.write_frames(str(destination), (64, 64), fps=24,
                                             output_params=["-threads", "1"])
        writer.send(None)
        for i in range(6):
            writer.send(np.full((64, 64, 3), i * 30, dtype=np.uint8))
        writer.close()
        return {"duration_seconds": 0.25, "seed": seed, "prompt": scene.narration}
    monkeypatch.setattr(render, "make_motion_clip", motion)
    image = tmp_path / "input.png"
    Image.new("RGB", (300, 300), "#74a164").save(image)
    board = Storyboard(title="Test video", scenes=[
        {"heading": "First", "narration": "One short sentence.", "visual_query": "trees"},
        {"heading": "Second", "narration": "Another short sentence.", "visual_query": "trees"},
    ])
    video, archive, _, stats = render.render_storyboard(board, "test", aspect, [str(image)] * 2,
                                                       ai_motion=ai_motion)
    reader = imageio_ffmpeg.read_frames(video)
    metadata = next(reader)
    assert metadata["size"] == dimensions
    assert abs(metadata["duration"] - 1.0) < 0.1
    assert len(list(reader)) == 24
    assert stats["scene_seconds"] == [0.5, 0.5]
    assert stats["ai_motion"] is ai_motion
    with zipfile.ZipFile(archive) as bundle:
        expected = {"storycut.mp4", "captions.srt", "storyboard.json", "credits.txt"}
        if ai_motion:
            expected.add("motion.json")
            assert len(json.loads(bundle.read("motion.json"))) == 2
            assert "AI-generated" in bundle.read("credits.txt").decode()
        assert set(bundle.namelist()) == expected
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


def test_motion_failure_is_not_silently_replaced_with_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(render, "narrate", lambda *_: (0.5, [(0.0, 0.5, "Test")]))
    def fail(*_):
        raise RuntimeError("GPU allowance exhausted")
    monkeypatch.setattr(render, "make_motion_clip", fail)
    image = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "green").save(image)
    from storycut.storyboard import EXAMPLE
    with pytest.raises(RuntimeError, match="GPU allowance exhausted"):
        render.render_storyboard(EXAMPLE, "test", "Portrait · 9:16", [str(image)] * 4,
                                  ai_motion=True)
    assert not list(render.OUTPUTS.iterdir())

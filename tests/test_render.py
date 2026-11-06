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
        expected = {"storycut.mp4", "captions.srt", "storyboard.json", "credits.txt", "timing.json"}
        if ai_motion:
            expected.add("motion.json")
            assert len(json.loads(bundle.read("motion.json"))) == 2
            assert "AI-generated" in bundle.read("credits.txt").decode()
        assert set(bundle.namelist()) == expected
        assert "00:00:00,500 --> 00:00:01,000" in bundle.read("captions.srt").decode()
        assert json.loads(bundle.read("storyboard.json"))["title"] == "Test video"


def test_failed_render_removes_partial_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(render, "narrate", lambda *_: (0.5, [(0.0, 0.5, "Test")]))
    def fail(*_):
        raise ValueError("Image unavailable")
    monkeypatch.setattr(render, "commons_visual", fail)
    from storycut.storyboard import EXAMPLE
    with pytest.raises(ValueError, match="Image unavailable"):
        render.render_storyboard(EXAMPLE, "test", "Portrait · 9:16", None)
    assert not list(render.OUTPUTS.iterdir())


def test_duration_mismatch_stops_before_images_or_motion(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(render, "narrate", lambda *_: (8.0, [(0.0, 7.8, "Test")]))
    def forbidden(*args):
        pytest.fail("A large narration mismatch must stop before images or GPU motion")
    monkeypatch.setattr(render, "commons_visual", forbidden)
    monkeypatch.setattr(render, "make_motion_clip", forbidden)
    from storycut.storyboard import EXAMPLE
    with pytest.raises(ValueError, match="Expand"):
        render.render_storyboard(EXAMPLE, "test", "Portrait · 9:16", None,
                                 ai_motion=True, target_seconds=60)
    assert not list(render.OUTPUTS.iterdir())


@pytest.mark.parametrize("target,ratio,ai_motion", [(30, 1.08, False), (45, 0.94, False), (60, 0.9, False), (60, 0.94, True)])
def test_export_matches_selected_duration_and_retimes_audio_and_subtitles(tmp_path, monkeypatch, target, ratio, ai_motion):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    # Keep this integration test inexpensive while exercising real FFmpeg audio/video.
    monkeypatch.setattr(render, "FORMATS", {"small": (160, 288)})
    duration = target * ratio / 2
    def speech(text, voice, destination):
        count = round((duration - 0.12) * 24000)
        tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(count) / 24000)
        sf.write(destination, np.concatenate([tone, np.zeros(2880)]), 24000)
        return duration, [(0.0, duration - 0.12, text)]
    monkeypatch.setattr(render, "narrate", speech)
    def motion(source, scene, size, destination, seed):
        writer = imageio_ffmpeg.write_frames(str(destination), (64, 64), fps=24,
                                             output_params=["-threads", "1"])
        writer.send(None)
        for index in range(6):
            writer.send(np.full((64, 64, 3), index * 30, dtype=np.uint8))
        writer.close()
        return {"duration_seconds": 0.25, "seed": seed}
    monkeypatch.setattr(render, "make_motion_clip", motion)
    image = tmp_path / "input.png"
    Image.new("RGB", (160, 288), "green").save(image)
    board = Storyboard(title="Timed export", scenes=[
        {"heading": "First", "narration": "First scene.", "visual_query": "trees"},
        {"heading": "Second", "narration": "Last scene.", "visual_query": "trees"},
    ])
    video, archive, _, stats = render.render_storyboard(board, "test", "small", [str(image)] * 2,
                                                       target_seconds=target, ai_motion=ai_motion)
    reader = imageio_ffmpeg.read_frames(video)
    metadata = next(reader)
    assert abs(metadata["duration"] - target) < 0.15
    assert len(list(reader)) == target * 24
    assert stats["speech_tempo"] == pytest.approx(ratio)
    with zipfile.ZipFile(archive) as bundle:
        captions = bundle.read("captions.srt").decode()
        assert f"{render.srt_time(target / 2)} --> {render.srt_time(target - 0.12 / ratio)}" in captions
        assert json.loads(bundle.read("timing.json"))["target_seconds"] == target
    render.run_ffmpeg(["-i", video, "-vn", "-c:a", "pcm_s16le", "decoded.wav"], tmp_path)
    audio, rate = sf.read(tmp_path / "decoded.wav")
    assert abs(len(audio) / rate - target) < 0.15
    # Tempo must preserve pitch; ensure the end of the final narration survived.
    sample = audio[rate:2 * rate]
    dominant = np.fft.rfftfreq(len(sample), 1 / rate)[np.argmax(abs(np.fft.rfft(sample)))]
    assert abs(dominant - 440) < 2
    assert np.max(abs(audio[int((target - 0.3) * rate):int((target - 0.2) * rate)])) > 0.03


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


def test_many_static_captions_do_not_stall_and_switch_at_their_audio_times(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(render, "FORMATS", {"small": (160, 288)})
    colours = [(230, 30, 30), (30, 220, 30), (30, 30, 230), (220, 220, 30),
               (220, 30, 220), (30, 220, 220), (220, 220, 220)]
    def speech(text, voice, destination):
        sf.write(destination, np.zeros(15 * 24000), 24000)
        return 15.0, [(i * 15 / 7, (i + 1) * 15 / 7, str(i)) for i in range(7)]
    def overlay(path, size, heading, caption, *args, **kwargs):
        Image.new("RGBA", size, (*colours[int(caption)], 255)).save(path)
    monkeypatch.setattr(render, "narrate", speech)
    monkeypatch.setattr(render, "overlay", overlay)
    original_run = render.subprocess.run
    def bounded_run(*args, **kwargs):
        kwargs["timeout"] = 15
        return original_run(*args, **kwargs)
    monkeypatch.setattr(render.subprocess, "run", bounded_run)
    source = tmp_path / "image.png"
    Image.new("RGB", (160, 288), "black").save(source)
    board = Storyboard(title="Many captions", scenes=[
        {"heading": "Scene", "narration": "Sample narration.", "visual_query": "trees"}
        for _ in range(2)
    ])
    video, _, _, _ = render.render_storyboard(board, "test", "small", [str(source)] * 2, target_seconds=30)
    reader = imageio_ffmpeg.read_frames(video)
    next(reader)
    samples = {round((i + 0.5) * 15 / 7 * 24): colour for i, colour in enumerate(colours)}
    seen = 0
    for frame_number, frame in enumerate(reader):
        if frame_number in samples:
            pixel = np.frombuffer(frame, np.uint8).reshape(288, 160, 3)[144, 80]
            assert np.max(abs(pixel.astype(int) - samples[frame_number])) < 8
            seen += 1
    assert seen == 7

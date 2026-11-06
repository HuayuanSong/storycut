from types import SimpleNamespace

import imageio_ffmpeg
from PIL import Image

from storycut import motion
from storycut.storyboard import EXAMPLE


def test_image_and_storyboard_condition_a_bounded_clip(tmp_path, monkeypatch):
    calls = []
    def pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=[[kwargs["image"]] * motion.FRAMES])
    monkeypatch.setattr(motion, "load_motion", lambda: pipeline)
    image = tmp_path / "input.png"
    Image.new("RGBA", (600, 400), "green").save(image)
    destination = tmp_path / "motion.mp4"
    scene = EXAMPLE.scenes[0]
    metadata = motion.make_motion_clip(image, scene, (720, 1280), destination, 42)
    call, = calls
    assert call["image"].size == (288, 512)
    assert call["image"].mode == "RGB"
    assert scene.narration in call["prompt"]
    assert scene.visual_query in call["prompt"]
    assert call["num_frames"] == 97
    assert len(call["timesteps"]) == 8
    assert call["guidance_scale"] == 1.0
    assert metadata["seed"] == 42
    frames = imageio_ffmpeg.read_frames(str(destination))
    info = next(frames)
    assert info["size"] == (288, 512)
    assert len(list(frames)) == 97

"""Optional image-to-video generation; lease a GPU only for one scene at a time."""

import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock

import imageio_ffmpeg
import spaces
from PIL import Image, ImageOps

from storycut.storyboard import Scene

MODEL_ID = "Lightricks/LTX-Video"
MODEL_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
CHECKPOINT = "ltxv-2b-0.9.8-distilled.safetensors"
# The 2B checkpoint uses the same transformer/VAE architecture as 0.9.5.
# These provide configs, tokenizer and T5; both video components come from 0.9.8.
CONFIG_ID = "Lightricks/LTX-Video-0.9.5"
CONFIG_REVISION = "e58e28c39631af4d1468ee57a853764e11c1d37e"
TIMESTEPS = [1000.0, 993.7, 987.5, 981.2, 975.0, 909.4, 725.0, 421.9]
FRAMES = 97
FPS = 24
_lock = Lock()
_startup_failed = False
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_motion():
    import torch
    from diffusers import AutoencoderKLLTXVideo, LTXConditionPipeline, LTXVideoTransformer3DModel
    from huggingface_hub import hf_hub_download
    from storycut.models import device

    target = device()
    dtype = torch.float32 if target == "cpu" else torch.bfloat16
    checkpoint = hf_hub_download(MODEL_ID, CHECKPOINT, revision=MODEL_REVISION)
    options = dict(config=CONFIG_ID, config_revision=CONFIG_REVISION, dtype=dtype)
    transformer = LTXVideoTransformer3DModel.from_single_file(
        checkpoint, subfolder="transformer", **options)
    vae = AutoencoderKLLTXVideo.from_single_file(checkpoint, subfolder="vae", **options)
    pipeline = LTXConditionPipeline.from_pretrained(
        CONFIG_ID, revision=CONFIG_REVISION, transformer=transformer, vae=vae, dtype=dtype,
    ).to(target)
    pipeline.vae.enable_tiling()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def prepare_motion() -> bool:
    """Preload for ZeroGPU without taking the photo renderer down on load failure."""
    global _startup_failed
    try:
        load_motion()
    except Exception:
        _startup_failed = True
        logger.exception("Optional video model could not load; photo mode remains available")
    return not _startup_failed


def scene_prompt(scene: Scene) -> str:
    return (
        f"A continuous cinematic shot of {scene.visual_query}. {scene.heading}. "
        f"The story of this shot: {scene.narration} "
        "Animate the supplied photograph with gentle, natural movement and a slow camera move. "
        "Preserve its main subject, composition, appearance and lighting. "
        "One steady shot, realistic motion, no cuts, no new objects, no text or subtitles."
    )


@spaces.GPU(duration=60)
def _generate_frames(image: Image.Image, prompt: str, seed: int):
    import torch

    with _lock, torch.inference_mode():
        pipeline = load_motion()
        result = pipeline(
            image=image, prompt=prompt, width=image.width, height=image.height,
            num_frames=FRAMES, frame_rate=FPS, num_inference_steps=len(TIMESTEPS),
            timesteps=TIMESTEPS, guidance_scale=1.0, decode_timestep=0.05,
            decode_noise_scale=0.025, generator=torch.Generator("cpu").manual_seed(seed),
        )
    return result.frames[0]


def make_motion_clip(image_path: Path, scene: Scene, size: tuple[int, int],
                     destination: Path, seed: int) -> dict:
    if _startup_failed:
        raise ValueError("AI motion is temporarily unavailable. Turn it off to make a photo video.")
    width, height = (288, 512) if size[1] > size[0] else (512, 288)
    with Image.open(image_path) as source:
        image = ImageOps.fit(source.convert("RGB"), (width, height))
    prompt = scene_prompt(scene)
    # Image decoding and encoding happen outside the GPU lease.
    frames = _generate_frames(image, prompt, seed)
    if len(frames) != FRAMES:
        raise RuntimeError("The video model returned an incomplete clip.")
    writer = imageio_ffmpeg.write_frames(
        str(destination), (width, height), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
        output_params=["-threads", "2", "-preset", "veryfast"],
    )
    try:
        writer.send(None)
        for frame in frames:
            writer.send(frame.convert("RGB").tobytes())
    finally:
        writer.close()
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "checkpoint": CHECKPOINT,
            "prompt": prompt, "seed": seed, "width": width, "height": height,
            "frames": FRAMES, "fps": FPS, "duration_seconds": FRAMES / FPS,
            "timesteps": TIMESTEPS}

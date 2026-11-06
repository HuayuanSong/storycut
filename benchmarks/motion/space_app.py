"""Private ZeroGPU benchmark. Does not modify the live Storycut app."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr
import imageio_ffmpeg
import spaces
import torch
from PIL import Image, ImageOps

# Keep the same writer resident as in production when measuring memory and speed.
from storycut.models import load_writer
from storycut.motion import FRAMES, FPS, TIMESTEPS, load_motion
from benchmarks.motion.fastwan_scheduler import FastWanDMDScheduler

BACKEND = os.environ["STORYCUT_BENCH_MODEL"]
WAN_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
WAN_REVISION = "3e187042a324f6f5fb68fd22110a78725253de8f"
load_writer()
if BACKEND == "ltx":
    pipeline = load_motion()
elif BACKEND == "fastwan":
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    vae = AutoencoderKLWan.from_pretrained(WAN_ID, revision=WAN_REVISION,
                                          subfolder="vae", dtype=torch.float32)
    pipeline = WanImageToVideoPipeline.from_pretrained(
        WAN_ID, revision=WAN_REVISION, vae=vae, dtype=torch.bfloat16,
        scheduler=FastWanDMDScheduler(),
    ).to("cuda")
    # Four-second 512x288 clips fit in this allocation without spatial tiling.
    # Keep a separate untiled trial so memory-saving tile overhead is not mistaken
    # for an inherent limit of the model.
    if os.getenv("STORYCUT_BENCH_TILING", "0") == "1":
        pipeline.vae.enable_tiling()
    else:
        pipeline.vae.disable_tiling()
    pipeline.set_progress_bar_config(disable=True)
else:
    raise ValueError("Unknown benchmark backend")


@spaces.GPU(duration=60)
def infer(image, prompt, seed):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = torch.Generator("cpu").manual_seed(seed)
    marks = []
    def after_step(pipe, index, timestep, tensors):
        torch.cuda.synchronize()
        marks.append(time.perf_counter() - started)
        return tensors
    common = dict(image=image, prompt=prompt, width=image.width, height=image.height,
                  num_frames=FRAMES, guidance_scale=1.0, generator=generator, output_type="pil")
    with torch.inference_mode():
        if BACKEND == "ltx":
            output = pipeline(**common, frame_rate=FPS, num_inference_steps=8,
                              timesteps=TIMESTEPS, decode_timestep=0.05, decode_noise_scale=0.025)
        else:
            pipeline.scheduler.generator = generator
            output = pipeline(**common, num_inference_steps=3, callback_on_step_end=after_step)
    torch.cuda.synchronize()
    metrics = {"backend": BACKEND, "gpu_body_seconds": time.perf_counter() - started,
               "vae_tiling": pipeline.vae.use_tiling, "step_end_seconds": marks,
               "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
               "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
               "device": torch.cuda.get_device_name(),
               "device_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
               "torch": torch.__version__, "frames": FRAMES, "fps": FPS,
               "seed": seed, "width": image.width, "height": image.height}
    return output.frames[0], metrics


def generate(image_path, prompt, seed, portrait):
    if not image_path or not isinstance(prompt, str) or not 1 <= len(prompt) <= 1500:
        raise gr.Error("Supply one image and a bounded prompt.")
    seed = int(seed)
    size = (288, 512) if portrait else (512, 288)
    with Image.open(image_path) as source:
        image = ImageOps.fit(source.convert("RGB"), size)
    started = time.perf_counter()
    frames, metrics = infer(image, prompt, seed)
    metrics["lease_wall_seconds"] = time.perf_counter() - started
    assert len(frames) == FRAMES
    job = Path(tempfile.mkdtemp(prefix="motion-bench-"))
    destination = job / "clip.mp4"
    writer = imageio_ffmpeg.write_frames(str(destination), size, fps=FPS, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
        output_params=["-threads", "2", "-preset", "veryfast"])
    try:
        writer.send(None)
        for frame in frames:
            writer.send(frame.convert("RGB").tobytes())
    finally:
        writer.close()
    metrics["server_seconds"] = time.perf_counter() - started
    (job / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return str(destination), metrics


with gr.Blocks(delete_cache=(3600, 86400)) as demo:
    gr.Markdown(f"Private motion benchmark: {BACKEND}")
    source = gr.Image(type="filepath")
    prompt = gr.Textbox()
    seed = gr.Number(value=42, precision=0)
    portrait = gr.Checkbox(value=True)
    run = gr.Button("Run benchmark")
    clip = gr.Video()
    metrics = gr.JSON()
    run.click(generate, [source, prompt, seed, portrait], [clip, metrics],
              api_name="benchmark", concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()

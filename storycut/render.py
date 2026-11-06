"""CPU assembly with optional generated motion, speech timings, and export bundle."""

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

from storycut.media import commons_visual, upload_visual
from storycut.motion import make_motion_clip
from storycut.speech import narrate
from storycut.storyboard import Storyboard
from storycut.timing import FPS, plan_timing, validate_target

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FORMATS = {"Portrait · 9:16": (720, 1280), "Landscape · 16:9": (1280, 720)}


def font(size: int, bold: bool = False):
    face = ImageFont.truetype(str(ROOT / "assets/Figtree.ttf"), size)
    face.set_variation_by_axes([700 if bold else 400])
    return face


def wrap(text: str, face, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if face.getlength(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        # Split very long unbroken tokens as well.
        while face.getlength(current) > width and len(current) > 1:
            cut = len(current) - 1
            while cut > 1 and face.getlength(current[:cut]) > width:
                cut -= 1
            lines.append(current[:cut])
            current = current[cut:]
    return [*lines, current] if current else lines


def overlay(path: Path, size: tuple[int, int], heading: str, caption: str, index: int, count: int,
            ai_motion: bool = False):
    width, height = size
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    # Transparent-to-ink gradients keep headings and subtitles legible on any photo.
    for y in range(height):
        opacity = max(0, 155 * (1 - y / (height * 0.4)),
                      220 * (y / height - 0.48) / 0.52)
        draw.line((0, y, width, y), fill=(10, 20, 27, min(230, int(opacity))))
    margin = 54
    draw.rounded_rectangle((margin, 50, margin + 92, 88), radius=12, fill="#e4f6b2")
    draw.text((margin + 15, 56), f"{index:02d} / {count:02d}", font=font(21, True), fill="#18262b")
    if ai_motion:
        draw.text((width - margin, 60), "AI-GENERATED VISUALS", anchor="ra", font=font(19), fill="white")
    heading_face = font(42 if height > width else 38, True)
    for row, line in enumerate(wrap(heading, heading_face, width - margin * 2)):
        draw.text((margin, 112 + row * 53), line, font=heading_face, fill="white")
    if caption:
        caption_face = font(34 if height > width else 32)
        lines = wrap(caption, caption_face, width - margin * 2)
        y = height - 100 - len(lines) * 46
        for line in lines:
            draw.text((margin, y), line, font=caption_face, fill="white", stroke_width=1,
                      stroke_fill=(10, 20, 27, 70))
            y += 46
    image.save(path)


def run_ffmpeg(arguments: list[str], cwd: Path):
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                             "-nostdin", "-y", *arguments], cwd=cwd,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Video encoding failed. " + result.stderr[-700:])


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remaining = divmod(milliseconds, 3_600_000)
    minutes, remaining = divmod(remaining, 60_000)
    seconds, milliseconds = divmod(remaining, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def clean_old_exports():
    OUTPUTS.mkdir(exist_ok=True)
    for path in OUTPUTS.glob("video-*"):
        if path.is_dir() and time.time() - path.stat().st_mtime > 24 * 3600:
            shutil.rmtree(path)


def render_storyboard(board: Storyboard, voice: str, aspect: str, uploads: list[str] | None,
                      progress=lambda *_: None, *, ai_motion: bool = False,
                      target_seconds: int | None = None
                      ) -> tuple[str, str, list[tuple[str, str]], dict]:
    if aspect not in FORMATS:
        raise ValueError("Choose portrait or landscape.")
    if uploads and len(uploads) != len(board.scenes):
        raise ValueError(f"Upload exactly {len(board.scenes)} images, one per scene in order, or leave uploads empty.")
    if not isinstance(ai_motion, bool):
        raise ValueError("Choose whether to enable AI motion.")
    if target_seconds is not None:
        validate_target(target_seconds)
    clean_old_exports()
    job = Path(tempfile.mkdtemp(prefix="video-", dir=OUTPUTS))
    width, height = FORMATS[aspect]
    credits, thumbnails, caption_rows, timings = [], [], [], []
    motion_details = []
    used, offset = set(), 0.0
    start = time.monotonic()
    try:
        speech = []
        for index, scene in enumerate(board.scenes):
            progress(index / len(board.scenes) * 0.15,
                     f"Checking narration length · scene {index+1}/{len(board.scenes)}")
            speech.append(narrate(scene.narration, voice, job / f"scene-{index:02d}.wav"))
        # Reject large mismatches before image requests, GPU motion or encoding.
        plan = plan_timing([duration for duration, _ in speech], target_seconds)
        visuals = []
        for index, scene in enumerate(board.scenes):
            progress(0.15 + index / len(board.scenes) * 0.15, f"Finding image {index+1}/{len(board.scenes)}")
            destination = job / f"scene-{index:02d}.jpg"
            visuals.append(upload_visual(uploads[index], destination) if uploads
                           else commons_visual(scene.visual_query, destination, used))
        for index, scene in enumerate(board.scenes):
            prefix = f"scene-{index:02d}"
            visual = visuals[index]
            credits.append(visual.credit())
            thumbnails.append((str(visual.path), f"{index+1}. {scene.heading}"))
            with Image.open(visual.path) as image:
                ImageOps.fit(image, (width, height)).save(job / f"{prefix}-crop.jpg", quality=94)
            captions = [(begin / plan.tempo, end / plan.tempo, text)
                        for begin, end, text in speech[index][1]]
            frame_count = plan.scene_frames[index]
            video_duration = frame_count / FPS
            if ai_motion:
                progress(0.3 + (index + 0.1) / len(board.scenes) * 0.65,
                         f"Scene {index+1}/{len(board.scenes)} · generating AI motion (GPU queue may apply)")
                details = make_motion_clip(visual.path, scene, (width, height),
                                           job / f"{prefix}-motion.mp4", secrets.randbelow(2**31))
                motion_details.append(details)
                arguments = ["-i", f"{prefix}-motion.mp4", "-i", f"{prefix}.wav"]
                # Play each short generated shot once, adapting its speed to the narration.
                # Pad the final frame for timestamp rounding before trimming to exact duration.
                playback_scale = video_duration / details["duration_seconds"]
                filters = [f"[0:v]setpts={playback_scale:.9f}*(PTS-STARTPTS),fps={FPS},"
                           f"scale={width}:{height}:flags=lanczos,setsar=1,"
                           f"tpad=stop_mode=clone:stop_duration={video_duration},"
                           f"trim=duration={video_duration}[bg0]"]
            else:
                arguments = ["-loop", "1", "-framerate", str(FPS), "-i", f"{prefix}-crop.jpg",
                             "-i", f"{prefix}.wav"]
                filters = [f"[0:v]zoompan=z='1+0.045*on/{frame_count}':"
                           f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={width}x{height}:fps={FPS}[bg0]"]
            # One PNG per spoken phrase. Each caption is tied to its own generated audio.
            # The last overlay also covers the tiny trailing silence/frame rounding.
            for number, (begin, end, caption) in enumerate(captions):
                name = f"{prefix}-caption-{number}.png"
                overlay(job / name, (width, height), scene.heading, caption, index + 1,
                        len(board.scenes), ai_motion=ai_motion)
                # Overlay repeats the final image by default. Decode each PNG
                # once; looping every caption creates unnecessary input queues.
                arguments += ["-i", name]
                visible_end = captions[number+1][0] if number+1 < len(captions) else video_duration
                filters.append(f"[bg{number}][{number+2}:v]overlay=enable='gte(t,{begin})*lt(t,{visible_end})'[bg{number+1}]")
                caption_rows.append((offset + begin, offset + end, caption))
            filters.append(f"[bg{len(captions)}]format=yuv420p[v]")
            tempo_filter = f"atempo={plan.tempo:.12f}," if plan.tempo != 1.0 else ""
            filters.append(f"[1:a]{tempo_filter}apad,atrim=duration={video_duration},asetpts=PTS-STARTPTS[a]")
            arguments += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
                          "-map", "[v]", "-map", "[a]", "-t", str(video_duration),
                          "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "23",
                          "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", f"{prefix}.mkv"]
            progress(0.3 + (index + 0.5) / len(board.scenes) * 0.65, f"Scene {index+1}/{len(board.scenes)} · rendering")
            run_ffmpeg(arguments, job)
            offset += video_duration
            timings.append(video_duration)
        progress(0.97, "Joining scenes and preparing your downloads")
        # Encode AAC once after joining; per-scene AAC padding otherwise accumulates drift.
        (job / "concat.txt").write_text("".join(
            f"file 'scene-{i:02d}.mkv'\nduration {duration:.9f}\n" for i, duration in enumerate(timings)))
        run_ffmpeg(["-f", "concat", "-safe", "1", "-i", "concat.txt", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k",
                    "-metadata", f"title={board.title}", "-movflags", "+faststart", "storycut.mp4"], job)
        (job / "captions.srt").write_text("\n".join(
            f"{i+1}\n{srt_time(begin)} --> {srt_time(end)}\n{text}\n"
            for i, (begin, end, text) in enumerate(caption_rows)), encoding="utf-8")
        (job / "storyboard.json").write_text(board.model_dump_json(indent=2), encoding="utf-8")
        credit_text = "Visual credits\n\nImages were cropped and animated. Keep these credits with the video when sharing.\n\n"
        if ai_motion:
            credit_text += ("Visuals are AI-generated interpretations of the credited source images, using "
                            "Lightricks LTX-Video 2B 0.9.8 distilled. They are not documentary footage.\n"
                            "Model terms: https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt\n\n")
        for i, credit in enumerate(credits):
            credit_text += f"Scene {i+1}: {credit['title']}\nCreator: {credit['creator']}\nLicense: {credit['license']} {credit['license_url']}\nSource: {credit['source_url']}\n\n"
        (job / "credits.txt").write_text(credit_text, encoding="utf-8")
        if ai_motion:
            (job / "motion.json").write_text(json.dumps(motion_details, indent=2), encoding="utf-8")
        timing_details = {"target_seconds": target_seconds, "narration_seconds": round(plan.narration_seconds, 3),
                          "speech_tempo": plan.tempo, "scene_seconds": timings,
                          "duration_seconds": round(offset, 3)}
        (job / "timing.json").write_text(json.dumps(timing_details, indent=2), encoding="utf-8")
        with zipfile.ZipFile(job / "storycut-bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in ("storycut.mp4", "captions.srt", "storyboard.json", "credits.txt", "timing.json"):
                bundle.write(job / name, name)
            if ai_motion:
                bundle.write(job / "motion.json", "motion.json")
        for path in job.glob("scene-*"):
            if path.name.endswith(".jpg") and "-crop" not in path.name:
                continue
            path.unlink()
        (job / "concat.txt").unlink()
        progress(1, "Your video is ready")
        stats = {"duration_seconds": round(offset, 2), "render_seconds": round(time.monotonic()-start, 2),
                 "scene_seconds": timings, "credits": credits, "ai_motion": ai_motion,
                 "target_seconds": target_seconds, "narration_seconds": plan.narration_seconds,
                 "speech_tempo": plan.tempo}
        return str(job / "storycut.mp4"), str(job / "storycut-bundle.zip"), thumbnails, stats
    except BaseException:
        shutil.rmtree(job, ignore_errors=True)
        raise

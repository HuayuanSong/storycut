"""CPU speech synthesis with a cached ONNX model and bounded sentence chunks."""

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro

VOICES = {"Heart · American": ("af_heart", "en-us"),
          "Sarah · American": ("af_sarah", "en-us"),
          "Emma · British": ("bf_emma", "en-gb"),
          "George · British": ("bm_george", "en-gb")}


@lru_cache(maxsize=1)
def load_voice() -> Kokoro:
    # Public mirror of the original Kokoro ONNX release; no paid inference API.
    repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
    model_path = hf_hub_download(repo, "onnx/model.onnx", revision="1939ad2a8e416c0acfeecc08a694d14ef25f2231")
    # kokoro-onnx expects the release's packed NumPy voice bank.
    from storycut.download import cached_release
    voices_path = cached_release("voices-v1.0.bin")
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(2, os.cpu_count() or 1))
    session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, str(voices_path))


def chunks(text: str) -> list[str]:
    """Break at sentence boundaries; long sentences become short caption phrases."""
    phrases = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for phrase in phrases:
        words = phrase.split()
        result.extend(" ".join(words[i:i+16]) for i in range(0, len(words), 16))
    return result


@lru_cache(maxsize=16)
def synthesize(text: str, voice: str) -> tuple[np.ndarray, int, tuple[tuple[float, float, str], ...]]:
    """Reuse bounded, in-memory speech between duration checks and rendering."""
    if voice not in VOICES:
        raise ValueError("Choose a voice from the list.")
    voice_id, lang = VOICES[voice]
    engine = load_voice()
    pieces, captions, elapsed = [], [], 0.0
    sample_rate = 24000
    for phrase in chunks(text):
        samples, sample_rate = engine.create(phrase, voice=voice_id, speed=1.0, lang=lang)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Speech generation returned invalid audio. Please try again.")
        duration = len(samples) / sample_rate
        captions.append((elapsed, elapsed + duration, phrase))
        pieces.extend((samples, np.zeros(round(sample_rate * 0.12), dtype=np.float32)))
        elapsed += duration + 0.12
    if not pieces or elapsed > 40:
        raise ValueError("This scene is too long. Shorten its narration and try again.")
    audio = np.concatenate(pieces)
    audio.setflags(write=False)
    return audio, sample_rate, tuple(captions)


def narrate(text: str, voice: str, destination: Path) -> tuple[float, list[tuple[float, float, str]]]:
    audio, sample_rate, captions = synthesize(text, voice)
    sf.write(destination, audio, sample_rate)
    return len(audio) / sample_rate, list(captions)


def measure_narration(texts: list[str], voice: str, destination: Path | None = None) -> list[float]:
    """Measure real speech; optionally save an unadjusted listening preview."""
    speech = [synthesize(text, voice) for text in texts]
    if destination is not None:
        rates = {rate for _, rate, _ in speech}
        if len(rates) != 1:
            raise ValueError("Narration sample rates did not match. Please try again.")
        sf.write(destination, np.concatenate([audio for audio, _, _ in speech]), rates.pop())
    return [len(audio) / rate for audio, rate, _ in speech]

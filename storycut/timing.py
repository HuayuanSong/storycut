"""Bounded narration pacing and frame-aligned scene durations."""

import math
from dataclasses import dataclass

FPS = 24
LENGTHS = (30, 45, 60)
MIN_TEMPO = 0.9
MAX_TEMPO = 1.1
MAX_SCENE_PAUSE = 0.35


def validate_target(seconds: int) -> None:
    if isinstance(seconds, bool) or seconds not in LENGTHS:
        raise ValueError("Choose a target length of 30, 45, or 60 seconds.")


def word_budget(seconds: int) -> int:
    validate_target(seconds)
    # Initial estimate only: measured Kokoro narration is the final authority.
    return round(seconds * 3.3)


class DurationMismatch(ValueError):
    """The script needs revision to fit without excessive pacing changes."""


@dataclass(frozen=True)
class TimingPlan:
    tempo: float
    scene_frames: tuple[int, ...]
    narration_seconds: float
    target_seconds: int | None

    @property
    def scene_seconds(self) -> list[float]:
        return [frames / FPS for frames in self.scene_frames]


def plan_timing(durations: list[float], target_seconds: int | None = None) -> TimingPlan:
    if not durations or any(not math.isfinite(d) or d <= 0 for d in durations):
        raise ValueError("Narration did not produce a valid duration. Please try again.")
    total = sum(durations)
    tempo, pause = 1.0, 0.0
    if target_seconds is not None:
        validate_target(target_seconds)
        tempo = max(MIN_TEMPO, min(MAX_TEMPO, total / target_seconds))
        remaining = target_seconds - total / tempo
        if remaining < -1e-6 or remaining > MAX_SCENE_PAUSE * len(durations) + 1e-6:
            action = "Expand" if total < target_seconds else "Shorten"
            raise DurationMismatch(
                f"Narration is {total:.1f}s; your target is {target_seconds}s. "
                f"{action} the script to fit, edit the narration, or choose another length. "
                "No video was generated: this gap needs more than a small pacing adjustment."
            )
        pause = max(0.0, remaining) / len(durations)
    # Round each scene up so its final spoken samples are retained. With at most
    # six scenes the complete export is less than 0.25s above the selected target.
    frames = tuple(math.ceil((duration / tempo + pause) * FPS - 1e-8) for duration in durations)
    return TimingPlan(tempo, frames, total, target_seconds)

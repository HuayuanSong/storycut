import math

import pytest

from storycut.timing import DurationMismatch, FPS, plan_timing


@pytest.mark.parametrize("target", [30, 45, 60])
@pytest.mark.parametrize("ratio", [0.91, 1.0, 1.08, 1.1])
def test_small_mismatches_fit_without_cutting_speech(target, ratio):
    durations = [target * ratio * share for share in (0.19, 0.24, 0.26, 0.31)]
    plan = plan_timing(durations, target)
    assert 0.9 <= plan.tempo <= 1.1
    assert target - 1e-6 <= sum(plan.scene_seconds) < target + len(durations) / FPS
    for natural, fitted in zip(durations, plan.scene_seconds, strict=True):
        assert fitted >= natural / plan.tempo - 1e-6


def test_brief_pauses_can_close_a_small_remaining_gap():
    durations = [12.2, 12.6, 13.7, 14.5]  # 53s -> 60s, at 0.9x with <0.35s pauses.
    plan = plan_timing(durations, 60)
    assert plan.tempo == 0.9
    for natural, fitted in zip(durations, plan.scene_seconds, strict=True):
        assert 0 <= fitted - natural / plan.tempo < 0.35 + 1 / FPS


@pytest.mark.parametrize("durations", [[8, 8, 9, 10], [18, 20, 21, 21]])
def test_large_mismatch_requires_script_revision(durations):
    with pytest.raises(DurationMismatch, match="No video was generated"):
        plan_timing(durations, 60)


@pytest.mark.parametrize("durations", [[], [0], [-1], [math.nan], [math.inf]])
def test_invalid_audio_duration_is_rejected(durations):
    with pytest.raises(ValueError):
        plan_timing(durations, 30)


@pytest.mark.parametrize("target", [True, 0, 35, "60", float("nan")])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        plan_timing([15, 15], target)


def test_existing_natural_duration_api_remains_available():
    plan = plan_timing([0.5, 1.01])
    assert plan.tempo == 1.0
    assert plan.scene_frames == (12, 25)

import pytest

from storycut import models
from storycut.storyboard import EXAMPLE


def test_empty_optional_gradio_notes_are_normalized_before_gpu_lease(monkeypatch):
    received = []
    def generate(*args):
        received.append(args)
        return EXAMPLE
    monkeypatch.setattr(models, "_generate_storyboard", generate)
    assert models.write_storyboard("Walking", None, "Clear & curious", 30) == EXAMPLE
    assert received[0][1] == ""


@pytest.mark.parametrize("topic", [None, "", " " * 5, "x" * 501])
def test_invalid_brief_does_not_lease_gpu(monkeypatch, topic):
    def forbidden(*_):
        pytest.fail("Invalid input must not request a GPU")
    monkeypatch.setattr(models, "_generate_storyboard", forbidden)
    with pytest.raises(ValueError):
        models.write_storyboard(topic, None, "Clear & curious", 30)


def board_with_words(count):
    return EXAMPLE.model_copy(update={"scenes": [
        scene.model_copy(update={"narration": "word " * count}) for scene in EXAMPLE.scenes
    ]})


def test_short_draft_gets_one_bounded_correction(monkeypatch):
    answers = iter([board_with_words(10), board_with_words(39)])
    calls = []
    monkeypatch.setattr(models, "load_writer", lambda: (None, None))
    def generate(messages, *_):
        calls.append(list(messages))
        return next(answers).model_dump_json()
    monkeypatch.setattr(models, "generate_text", generate)
    result = models.write_storyboard("Walking", None, "Clear & curious", 60)
    assert result.word_count == 196
    assert len(calls) == 2
    assert "40 spoken words" in calls[1][-1]["content"]


def test_failed_correction_keeps_the_valid_draft(monkeypatch):
    answers = iter([board_with_words(20).model_dump_json(), "not JSON"])
    monkeypatch.setattr(models, "load_writer", lambda: (None, None))
    monkeypatch.setattr(models, "generate_text", lambda *_: next(answers))
    assert models.write_storyboard("Walking", None, "Clear & curious", 60).word_count == 80


def test_good_draft_does_not_use_an_extra_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "load_writer", lambda: (None, None))
    def generate(*_):
        calls.append(True)
        return board_with_words(37).model_dump_json()
    monkeypatch.setattr(models, "generate_text", generate)
    assert models.write_storyboard("Walking", None, "Clear & curious", 45).word_count == 148
    assert len(calls) == 1


def test_revision_keeps_user_metadata_and_uses_measured_speech_rate(monkeypatch):
    captured = []
    monkeypatch.setattr(models, "load_writer", lambda: (None, None))
    candidate = board_with_words(17)
    candidate.title = "Unwanted title"
    for scene in candidate.scenes:
        scene.heading = "Unwanted heading"
        scene.visual_query = "Unwanted subject"
    def generate(messages, *_):
        captured.append(messages)
        return candidate.model_dump_json()
    monkeypatch.setattr(models, "generate_text", generate)
    original = board_with_words(20)
    revised = models.revise_storyboard(original, 60, 32, "Keep these facts.")
    assert "70 spoken words" in captured[0][0]["content"]
    assert "Keep these facts." in captured[0][1]["content"]
    assert revised.title == original.title
    assert revised.scenes[0].heading == original.scenes[0].heading
    assert revised.scenes[0].visual_query == original.scenes[0].visual_query
    assert revised.word_count == 148
    assert original.word_count == 80
    assert revised.scenes[0].narration.startswith(original.scenes[0].narration)


def test_overlong_additions_keep_user_text_and_only_include_complete_sentences():
    original = board_with_words(40)
    candidate = EXAMPLE.model_copy(update={"scenes": [
        scene.model_copy(update={"narration": "A useful extra detail. Another complete sentence. One more relevant example."})
        for scene in EXAMPLE.scenes
    ]})
    revised = models.combine_revision(original, candidate, True, 180)
    assert 160 < revised.word_count <= 180
    for before, after in zip(original.scenes, revised.scenes, strict=True):
        assert after.narration.startswith(before.narration)
        assert after.narration.endswith(".")

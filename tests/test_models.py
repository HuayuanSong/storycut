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

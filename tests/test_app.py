import gradio as gr
import pytest

import app
from storycut.storyboard import EXAMPLE


@pytest.mark.parametrize("change", ["narration", "voice", "seconds", "title"])
def test_stale_suggestion_cannot_overwrite_edits(change):
    voice, seconds, title = next(iter(app.VOICES)), 45, EXAMPLE.title
    rows = EXAMPLE.rows()
    revision = {"base": app.revision_key(EXAMPLE, voice, seconds), "rows": rows}
    if change == "narration":
        rows[0][1] = "An edit I want to keep."
    elif change == "voice":
        voice = "Emma · British"
    elif change == "seconds":
        seconds = 60
    else:
        title = "My new title"
    with pytest.raises(gr.Error, match="changed after"):
        app.accept_revision(title, rows, voice, seconds, revision)


def test_revision_requires_explicit_acceptance(monkeypatch):
    revised = EXAMPLE.model_copy(update={"scenes": [
        scene.model_copy(update={"narration": "A suggested change."}) for scene in EXAMPLE.scenes
    ]})
    monkeypatch.setattr(app, "measure_narration", lambda *_: [10, 10, 10, 10])
    monkeypatch.setattr(app, "revise_storyboard", lambda *_: revised)
    rows = EXAMPLE.rows()
    voice = next(iter(app.VOICES))
    panel, suggestions, message, state = app.suggest_revision(EXAMPLE.title, rows, voice, 45, None)
    assert rows == EXAMPLE.rows()
    assert suggestions[0][1] == "A suggested change."
    assert "current script is unchanged" in message
    applied, _, cleared, hidden = app.accept_revision(EXAMPLE.title, rows, voice, 45, state)
    assert applied == revised.rows()
    assert cleared is None and hidden["visible"] is False


def test_render_receives_the_selected_target(monkeypatch):
    received = []
    def render(*args, **kwargs):
        received.append(kwargs["target_seconds"])
        return "video", "bundle", [], {"duration_seconds": 60, "render_seconds": 2}
    monkeypatch.setattr(app, "render_storyboard", render)
    app.make_video(EXAMPLE.title, EXAMPLE.rows(), next(iter(app.VOICES)), "Portrait · 9:16", None, False, 60)
    assert received == [60]

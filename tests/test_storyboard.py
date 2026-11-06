import json

import pytest
from pydantic import ValidationError

from storycut.storyboard import EXAMPLE, Storyboard, from_editor, parse_generation


def test_fenced_model_output_becomes_editable_board():
    board = parse_generation("```json\n" + EXAMPLE.model_dump_json() + "\n```")
    assert from_editor(board.title, board.rows()) == EXAMPLE


@pytest.mark.parametrize("output", ["no JSON here", '{"title":"Incomplete"}', '{"title":"bad","scenes":[]}'])
def test_broken_generation_is_not_silently_replaced(output):
    with pytest.raises(ValueError):
        parse_generation(output)


def test_renderer_input_cannot_expand_without_bound():
    data = EXAMPLE.model_dump()
    data["scenes"][0]["narration"] = "word " * 150
    with pytest.raises(ValidationError):
        Storyboard.model_validate(data)
    with pytest.raises(ValueError):
        from_editor("Too many scenes", EXAMPLE.rows() * 2)


def test_editor_does_not_drop_missing_or_extra_columns():
    with pytest.raises(ValueError):
        from_editor("Broken", [["heading", "narration"], ["heading", "narration"]])


def test_unknown_model_fields_are_rejected():
    data = EXAMPLE.model_dump()
    data["scenes"][0]["execute"] = "anything"
    with pytest.raises(ValueError):
        parse_generation(json.dumps(data))

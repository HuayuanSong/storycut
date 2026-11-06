"""Validated data shared by generation, the editor, and rendering."""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    heading: str = Field(min_length=1, max_length=70)
    narration: str = Field(min_length=1, max_length=500)
    visual_query: str = Field(min_length=1, max_length=100)

    @field_validator("visual_query")
    @classmethod
    def single_search_subject(cls, value: str) -> str:
        # Models sometimes propose several shots separated by commas. Search one.
        subject = " ".join(value.split(",", 1)[0].split()[:6])
        if not subject:
            raise ValueError("Enter a concrete image search, such as 'forest path'.")
        return subject

    @field_validator("heading", "narration", "visual_query")
    @classmethod
    def readable_text(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
            raise ValueError("Text contains unsupported control characters.")
        return re.sub(r"\s+", " ", value).strip()


class Storyboard(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=100)
    scenes: list[Scene] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def limit_length(self):
        if sum(len(scene.narration.split()) for scene in self.scenes) > 180:
            raise ValueError("Keep the narration to 180 words or fewer across all scenes.")
        return self

    @property
    def word_count(self) -> int:
        return sum(len(scene.narration.split()) for scene in self.scenes)

    def rows(self) -> list[list[str]]:
        return [[s.heading, s.narration, s.visual_query] for s in self.scenes]


def parse_generation(text: str) -> Storyboard:
    """Accept a JSON object, including one inside a model's Markdown fence."""
    start = text.find("{")
    if start == -1:
        raise ValueError("The model did not return a storyboard. Please try again.")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
        return Storyboard.model_validate(payload)
    except (ValueError, TypeError) as exc:
        raise ValueError("The model returned an invalid storyboard. Please try again.") from exc


def from_editor(title: str, rows: list[list[str]]) -> Storyboard:
    if not isinstance(rows, list) or not 2 <= len(rows) <= 6:
        raise ValueError("Your storyboard needs between 2 and 6 scenes.")
    if any(not isinstance(row, list) or len(row) != 3 for row in rows):
        raise ValueError("Each scene needs a heading, narration, and image search.")
    return Storyboard(title=title, scenes=[dict(zip(
        ("heading", "narration", "visual_query"), row, strict=True
    )) for row in rows])


EXAMPLE = Storyboard(title="The quiet power of a walk", scenes=[
    Scene(heading="Step outside", narration="Some days, a change of scenery starts with a single step. Leave the screen behind and take a short walk.", visual_query="forest walking path"),
    Scene(heading="Notice the small things", narration="Look for the details you usually miss: light through the leaves, a familiar street, or the rhythm of your footsteps.", visual_query="sunlight forest leaves"),
    Scene(heading="Make room to think", narration="You do not need a destination or a perfect route. Give yourself a few minutes with nowhere urgent to be.", visual_query="park footpath"),
    Scene(heading="Come back refreshed", narration="Start small. Choose a nearby path, put on comfortable shoes, and see how a little movement changes your afternoon.", visual_query="green park landscape"),
])

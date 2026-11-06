from io import BytesIO

import pytest
from PIL import Image

from storycut.media import clean_text, permitted_license, save_image


def test_license_policy_excludes_sharealike_and_unknown_terms():
    assert permitted_license("Public domain", "")
    assert permitted_license("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
    assert not permitted_license("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/")
    assert not permitted_license("CC BY 4.0", "https://example.com/licenses/by/4.0/")


def test_metadata_html_is_plain_text():
    assert clean_text('<a href="https://example.com">A &amp; B</a>\n') == "A & B"


def test_uploaded_image_is_decoded_and_reencoded(tmp_path):
    source = BytesIO()
    Image.new("RGBA", (30, 40), "red").save(source, format="PNG")
    target = tmp_path / "photo.jpg"
    save_image(source.getvalue(), target)
    with Image.open(target) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (30, 40)
    with pytest.raises(OSError):
        save_image(b"not an image", target)

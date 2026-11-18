"""Exercise Gradio job completion during real browser edits, offline and without a GPU."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
from unittest.mock import patch
import zipfile

import imageio_ffmpeg
import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from storycut.storyboard import EXAMPLE


def check(root):
    entered, release = Event(), Event()
    holding = None

    def hold(name):
        nonlocal holding
        holding = name
        entered.clear()
        release.clear()

    def pause(name):
        if holding == name:
            entered.set()
            assert release.wait(30), f"Timed out waiting to release {name}"

    generated = EXAMPLE.model_copy(update={"title": "Fresh AI draft"})

    def writer(*_):
        pause("draft")
        return generated

    def measure(texts, voice, destination=None):
        pause("timing")
        if destination:
            sf.write(destination, np.zeros(2400), 24000)
        return [7.5] * len(texts)

    def revise(board, *_):
        pause("revision")
        if board.title == "Repeat-only suggestion":
            raise ValueError("The model did not add any new narration that fits.")
        return board.model_copy(update={"scenes": [
            scene.model_copy(update={"narration": f"{scene.narration} Notice the birds."}) for scene in board.scenes
        ]})

    video = root / "short.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=s=64x64:d=0.25", "-c:v", "libx264", str(video)], check=True)
    bundle = root / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(video, "short.mp4")
    images = [root / f"review-{i}.png" for i in range(4)]
    for image in images:
        Image.new("RGB", (64, 64), "green").save(image)

    def render(board, *_, target_seconds, **kwargs):
        pause("render")
        return str(video), str(bundle), [(str(images[0]), board.title)], {
            "duration_seconds": target_seconds, "render_seconds": 1,
        }

    with patch.multiple(app, write_storyboard=writer, measure_narration=measure, revise_storyboard=revise,
                        render_storyboard=render, OUTPUTS=root, clean_old_exports=lambda: None):
        demo = app.build_app().queue()
        _, url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True,
                                allowed_paths=[str(root)], css_paths=app.ROOT / "assets/style.css")
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=os.getenv("PLAYWRIGHT_EXECUTABLE_PATH"))
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.set_default_timeout(12_000)
                page.goto(url, wait_until="domcontentloaded")
                button = lambda name: page.get_by_role("button", name=name)
                idea = page.get_by_role("tab", name="01 · Start with an idea")
                edit = page.get_by_role("tab", name="02 · Shape your story")
                watch = page.get_by_role("tab", name="03 · Watch & download")
                title = page.get_by_role("textbox", name="Video title")
                example = button("Explore with an example")
                check_length = button("Check narration length")
                make = button("Make my video")
                motion = page.get_by_role("checkbox", name="Animate images with AI")
                file_input = page.locator('input[type="file"][multiple]').first
                target = lambda seconds: page.get_by_role("radio", name=str(seconds), exact=True).last

                example.click()
                expect(title).to_have_value(EXAMPLE.title)
                page.get_by_text("Use your own images · optional", exact=True).click()
                file_input.set_input_files([str(path) for path in images])
                filename = page.get_by_role("cell", name=images[0].name, exact=True).first
                expect(filename).to_be_visible()
                motion.check()
                idea.click()
                example.click()
                expect(filename).to_have_count(0)
                expect(motion).not_to_be_checked()

                # A replacement project and text entered after submitting a draft take precedence.
                idea.click()
                page.get_by_role("textbox", name="What is your video about?").fill("An orchard")
                hold("draft")
                button("Write my storyboard").click()
                assert entered.wait(8)
                example.click()
                title.fill("My newer edit")
                release.set()
                idea.click()
                expect(button("Write my storyboard")).to_be_enabled()
                edit.click()
                expect(title).to_have_value("My newer edit")

                # An ordinary completed draft also resets project-specific inputs.
                file_input.set_input_files([str(path) for path in images])
                expect(filename).to_be_visible()
                motion.check()
                idea.click()
                button("Write my storyboard").click()
                expect(title).to_have_value(generated.title)
                expect(filename).to_have_count(0)
                expect(motion).not_to_be_checked()

                # A cell can have unsaved keystrokes when a queued result arrives.
                hold("render")
                make.click()
                assert entered.wait(8)
                original = EXAMPLE.scenes[0].narration
                cell = next(node for node in page.get_by_text(original, exact=True).all() if node.is_visible())
                cell.dblclick()
                cell_editor = page.get_by_role("textbox", name="Edit cell")
                cell_editor.fill("Words I am still typing.")
                release.set()
                expect(make).to_be_enabled()
                expect(edit).to_have_attribute("aria-selected", "true")
                expect(cell_editor).to_have_value("Words I am still typing.")
                cell_editor.press("Tab")

                # A render and a timing check queued behind it both become stale.
                hold("render")
                make.click()
                assert entered.wait(8)
                check_length.click()
                expect(check_length).to_be_disabled()
                title.fill("A newer film")
                target(45).check()
                release.set()
                expect(make).to_be_enabled()
                expect(check_length).to_be_enabled()
                expect(edit).to_have_attribute("aria-selected", "true")
                expect(page.get_by_text("Narration or target changed.", exact=False)).to_be_visible()
                watch.click()
                expect(page.get_by_text("Your storyboard changed.", exact=False)).to_be_visible()
                expect(page.locator('a[href*="bundle.zip"]:visible')).to_have_count(0)
                edit.click()

                check_length.click()
                expand = button("Expand script to fit")
                expect(expand).to_be_enabled()
                expect(page.get_by_text("Target: 45 seconds.", exact=False)).to_be_visible()
                hold("revision")
                expand.click()
                assert entered.wait(8)
                title.fill("Keep my edit")
                release.set()
                expect(page.get_by_text("Preparing a suggested revision…", exact=True)).to_be_hidden()
                expect(button("Revise script to fit")).to_be_disabled()
                expect(button("Use this revision")).to_have_count(0)
                title.fill("Repeat-only suggestion")
                check_length.click()
                expect(expand).to_be_enabled()
                expand.click()
                expect(page.get_by_text("The model did not add any new narration that fits.", exact=False).first).to_be_visible()
                expect(expand).to_be_enabled()
                expect(button("Use this revision")).to_have_count(0)
                title.fill("Keep my edit")
                check_length.click()
                expect(expand).to_be_enabled()
                expand.click()
                accept = button("Use this revision")
                expect(accept).to_be_visible()
                file_input.set_input_files([str(path) for path in images])
                expect(filename).to_be_visible()
                accept.click()
                expect(page.get_by_text("Revision applied.", exact=False)).to_be_visible()
                expect(filename).to_be_visible()
                expect(title).to_have_value("Keep my edit")
                make.click()
                expect(watch).to_have_attribute("aria-selected", "true")
                expect(page.locator('a[href*="bundle.zip"]:visible').first).to_be_visible()
                expect(page.get_by_text("Your video is ready.", exact=True).first).to_be_visible()
                for scheme in ("light", "dark"):
                    themed = browser.new_page(color_scheme=scheme)
                    themed.goto(url, wait_until="domcontentloaded")
                    themed.get_by_role("button", name="Explore with an example").click()
                    field = themed.get_by_role("textbox", name="Video title")
                    contrast = field.evaluate("""element => {
                        let surface = element;
                        while (getComputedStyle(surface).backgroundColor === "rgba(0, 0, 0, 0)") surface = surface.parentElement;
                        function luminance(color) {
                            const rgb = color.match(/\\d+/g).slice(0, 3).map(Number).map(n => n / 255);
                            const [r,g,b] = rgb.map(v => v <= 0.04045 ? v/12.92 : ((v+0.055)/1.055)**2.4);
                            return 0.2126*r + 0.7152*g + 0.0722*b;
                        }
                        const [a,b] = [getComputedStyle(element).color, getComputedStyle(surface).backgroundColor]
                            .map(luminance).sort((x,y) => x-y);
                        return (b+0.05)/(a+0.05);
                    }""")
                    assert contrast >= 4.5, f"{scheme} title contrast was {contrast:.2f}:1"
                    themed.close()
                browser.close()
        finally:
            release.set()
            demo.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        check(Path(folder))
    print("PASS project reset, stale and queued jobs, current draft, revision recovery, download and theme contrast")

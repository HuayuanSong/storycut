"""Check scrolling in the public Space without requesting text or video generation.

Run with .venv/bin/python tests/check_scrolling.py. Set
PLAYWRIGHT_EXECUTABLE_PATH to use an existing Chrome installation.
"""

import os

from playwright.sync_api import expect, sync_playwright


SPACE_URL = "https://huggingface.co/spaces/huathedev/storycut"


def check_scrolling(browser, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    try:
        page.goto(SPACE_URL, wait_until="domcontentloaded", timeout=60_000)
        app = page.frame_locator('iframe[aria-label="Space app"]')
        app.get_by_role("button", name="Explore with an example", exact=True).click(timeout=60_000)
        edit_tab = app.get_by_role("tab", name="02 · Shape your story")
        expect(edit_tab).to_have_attribute("aria-selected", "true")
        render = app.get_by_role("button", name="Make my video", exact=False)

        def reach_render():
            # Wheel scrolling must suffice; scroll_into_view would conceal the
            # bug by forcibly scrolling the iframe despite scrolling="no".
            page.mouse.move(width // 2, height // 2)
            for _ in range(3):
                page.mouse.wheel(0, 5000)
                page.wait_for_timeout(200)
            expect(render).to_be_in_viewport()
            render.click(trial=True)  # Check it is clickable without rendering.

        def select_tab(name):
            tab = app.get_by_role("tab", name=name, exact=True)
            more_tabs = app.get_by_role("button", name="More tabs", exact=True)
            if more_tabs.is_visible():
                # Gradio moves overflowing tabs into a menu on narrow screens.
                more_tabs.click()
                tab.or_(app.get_by_role("button", name=name, exact=True)).click()
            else:
                tab.click()
            expect(tab).to_have_attribute("aria-selected", "true")

        reach_render()
        app.get_by_text("Use your own images · optional", exact=True).click()
        reach_render()
        app.get_by_role("checkbox", name="Animate images with AI", exact=False).check()
        reach_render()
        select_tab("01 · Start with an idea")
        select_tab("02 · Shape your story")
        reach_render()
        print(f"PASS {width}×{height}: editor, image uploads, motion details and tab switching")
    finally:
        page.close()


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=os.getenv("PLAYWRIGHT_EXECUTABLE_PATH"),
        )
        try:
            for width, height in [(1440, 900), (1280, 720), (390, 844)]:
                check_scrolling(browser, width, height)
        finally:
            browser.close()


if __name__ == "__main__":
    main()

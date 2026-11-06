"""Gradio entrypoint, ready for a Hugging Face ZeroGPU Space."""

import json
import logging
import os
from pathlib import Path
import time

import gradio as gr
from pydantic import ValidationError

from storycut.models import IS_SPACE, write_storyboard
from storycut.render import FORMATS, OUTPUTS, render_storyboard
from storycut.speech import VOICES
from storycut.storyboard import EXAMPLE, from_editor

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent


def loading_event(button, feedback, message):
    """Show feedback in the browser before a queued request reaches the server."""
    markup = ('<div class="job-status" role="status" aria-live="polite">'
              '<span class="job-spinner" aria-hidden="true"></span>'
              f'<span>{message}</span></div>')
    return button.click(
        fn=None, outputs=[button, feedback], queue=False,
        js=f'() => [{{__type__: "update", interactive: false}}, {json.dumps(markup)}]',
    )


def clear_loading(event, button, feedback):
    # Explicit failure handling also covers Gradio versions whose .then skips errors.
    for finish in (event.success, event.failure):
        finish(fn=None, outputs=[button, feedback], queue=False,
               js='() => [{__type__: "update", interactive: true}, ""]')


def draft(topic, notes, tone, seconds, progress=gr.Progress()):
    progress(0, desc="Writing your storyboard · first run may need to load the model")
    start = time.monotonic()
    try:
        board = write_storyboard(topic, notes, tone, int(seconds))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Storyboard generation failed")
        raise gr.Error("The script model could not finish. If your GPU allowance is exhausted, try later; you can still edit and render an existing storyboard.") from exc
    return (board.title, board.rows(),
            f"Draft ready · {board.word_count} words · generated in {time.monotonic()-start:.0f}s. Review the wording and facts before rendering.",
            gr.Tabs(selected="edit"))


def load_example():
    return EXAMPLE.title, EXAMPLE.rows(), "Example storyboard loaded. This is a prewritten demo; edit it or generate your own.", gr.Tabs(selected="edit")


def make_video(title, rows, voice, aspect, uploads, progress=gr.Progress()):
    try:
        board = from_editor(title, rows)
        video, bundle, gallery, stats = render_storyboard(
            board, voice, aspect, uploads,
            progress=lambda value, description: progress(value, desc=description),
        )
    except ValidationError as exc:
        raise gr.Error("Check the storyboard: every scene needs a heading, narration, and image search. Keep it under 180 spoken words.") from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Video render failed")
        raise gr.Error("The video could not be completed. Please retry, or use your own images if image search is unavailable.") from exc
    info = (f"**Your video is ready.** {stats['duration_seconds']:.0f} seconds · "
            f"{len(board.scenes)} scenes · rendered in {stats['render_seconds']:.0f}s. "
            "Download the bundle for your MP4, captions, storyboard, and image credits.")
    return video, bundle, gallery, info, gr.Tabs(selected="watch")


def build_app():
    with gr.Blocks(title="Storycut · An idea into a short film", analytics_enabled=False,
                   delete_cache=(3600, 86400)) as demo:
        with gr.Row(elem_id="masthead"):
            gr.HTML('<div class="brand"><span class="brand-mark">▰</span> storycut<span class="beta">EARLY PREVIEW</span></div>')
            if IS_SPACE:
                gr.LoginButton(size="sm", scale=0)
        gr.HTML('''<section class="intro"><p class="eyebrow">SMALL IDEA. YOUR NEXT SHORT.</p>
            <h1>Give your ideas<br><em>a little screen time.</em></h1>
            <p class="lede">Turn a topic into a narrated short video. Shape the story, choose a voice, and make it yours.</p>
            <div class="steps"><span>01 &nbsp; An idea</span><b>→</b><span>02 &nbsp; Your story</span><b>→</b><span>03 &nbsp; A finished video</span></div></section>''')
        with gr.Tabs(selected="idea", elem_id="workspace") as tabs:
            with gr.Tab("01 · Start with an idea", id="idea"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=340):
                        topic = gr.Textbox(label="What is your video about?", placeholder="A small idea, a useful explanation, a place worth visiting…", lines=3, max_lines=5)
                        with gr.Row():
                            tone = gr.Dropdown(["Clear & curious", "Warm & reflective", "Punchy & upbeat"], value="Clear & curious", label="Voice of the story")
                            seconds = gr.Radio([30, 45, 60], value=30, label="Target length · seconds")
                        with gr.Accordion("Add source notes · optional", open=False):
                            notes = gr.Textbox(label="Facts you want the script to use", placeholder="Paste a few notes or an excerpt you have permission to use. Links alone are not fetched.", lines=4)
                        write = gr.Button("Write my storyboard  →", variant="primary", size="lg")
                        write_feedback = gr.HTML("", elem_id="storyboard-progress", elem_classes="job-feedback")
                        example = gr.Button("Explore with an example", variant="secondary")
                    with gr.Column(scale=2, min_width=280):
                        gr.HTML('''<div class="idea-card"><div class="card-top"><span>YOUR MINI STUDIO</span><span>↗</span></div>
                            <div class="film-frame"><div class="sun"></div><div class="hill hill-back"></div><div class="hill"></div><div class="play-icon">▶</div><span class="frame-caption">A fresh perspective.</span></div>
                            <h3>A first draft.<br>Then your finishing touch.</h3><p>AI writes the script. You review every scene. We add narration, photographs, motion, and captions.</p></div>''')
                gr.Markdown("**English narration · Portrait or landscape · Downloadable MP4**", elem_id="features")
            with gr.Tab("02 · Shape your story", id="edit"):
                status = gr.Markdown("Start with an idea, or load the example to explore the editor.")
                title = gr.Textbox(label="Video title", placeholder="Your title")
                editor = gr.Dataframe(headers=["On-screen heading", "Spoken narration", "Image search"],
                                      datatype="str", type="array", row_count=4,
                                      column_count=3, wrap=True, interactive=True,
                                      column_widths=["22%", "53%", "25%"], show_row_numbers=True,
                                      label="Your storyboard · click any cell to edit", max_height=500)
                with gr.Row():
                    voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Narrator")
                    aspect = gr.Radio(list(FORMATS), value=next(iter(FORMATS)), label="Video format")
                with gr.Accordion("Use your own images · optional", open=False):
                    uploads = gr.File(label="One image for each scene, in scene order", file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                    gr.Markdown("Leave this empty to search Wikimedia Commons. Upload images you have permission to use.")
                render = gr.Button("Make my video  →", variant="primary", size="lg")
                render_feedback = gr.HTML("", elem_id="video-progress", elem_classes="job-feedback")
                gr.Markdown("Rendering uses your edited script. It does not ask the LLM to rewrite it. Actual duration follows the narration.")
            with gr.Tab("03 · Watch & download", id="watch"):
                result = gr.Markdown("Your finished video will appear here.")
                video = gr.Video(label="Your short video", interactive=False, height=520, buttons=["download"])
                bundle = gr.File(label="Download the full bundle · MP4 + captions + storyboard + credits", interactive=False)
                with gr.Accordion("Images used in this video", open=False):
                    gallery = gr.Gallery(label="Scene images", columns=4, height=230, interactive=False)
                gr.Markdown("Keep **credits.txt** with your video when sharing Commons images. Downloads are temporary; save anything you want to keep.")
        gr.HTML('<footer class="foot"><span>Made with open models. Finished by you.</span><span>Qwen · Kokoro · Wikimedia Commons</span></footer>')
        writing = loading_event(write, write_feedback,
                                "Writing your storyboard… A GPU may need to become available first.")
        writing = writing.then(draft, [topic, notes, tone, seconds], [title, editor, status, tabs],
                               concurrency_limit=1, concurrency_id="writer", api_name="draft",
                               show_progress="minimal", show_progress_on=[write_feedback])
        clear_loading(writing, write, write_feedback)
        example.click(load_example, outputs=[title, editor, status, tabs], queue=False, api_name=False)
        rendering = loading_event(render, render_feedback,
                                  "Making your video… Finding images, creating narration and assembling your scenes.")
        rendering = rendering.then(make_video, [title, editor, voice, aspect, uploads],
                                    [video, bundle, gallery, result, tabs], concurrency_limit=1,
                                    concurrency_id="renderer", api_name="render",
                                    show_progress="minimal", show_progress_on=[render_feedback])
        clear_loading(rendering, render, render_feedback)
        # An old video must never look like the output of newly edited settings.
        def stale():
            return None, None, [], "Your storyboard changed. Render again to see the updated video."
        for control in (title, editor, voice, aspect, uploads):
            control.change(stale, outputs=[video, bundle, gallery, result], queue=False, api_name=False)
    return demo


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    theme = gr.themes.Base(primary_hue="lime", secondary_hue="slate", neutral_hue="stone",
                           font=["Inter", "Arial", "sans-serif"], text_size="lg")
    build_app().queue(max_size=12).launch(
        server_name="0.0.0.0" if IS_SPACE else "127.0.0.1",
        server_port=int(os.getenv("PORT", "7865" if not IS_SPACE else "7860")),
        theme=theme, css_paths=ROOT / "assets/style.css", max_file_size="15mb",
        allowed_paths=[str(OUTPUTS)], footer_links=[], show_error=False,
    )

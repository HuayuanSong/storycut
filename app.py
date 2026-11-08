"""Gradio entrypoint, ready for a Hugging Face ZeroGPU Space."""

import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

import gradio as gr
from pydantic import ValidationError

from storycut.models import IS_SPACE, revise_storyboard, write_storyboard
from storycut.motion import prepare_motion
from storycut.render import FORMATS, OUTPUTS, clean_old_exports, render_storyboard
from storycut.speech import VOICES, measure_narration
from storycut.storyboard import EXAMPLE, MAX_WORDS, from_editor
from storycut.timing import DurationMismatch, LENGTHS, plan_timing, validate_target, word_budget

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent
MOTION_AVAILABLE = prepare_motion() if IS_SPACE else True


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
    status = (f"Draft ready · {board.word_count} words · generated in {time.monotonic()-start:.0f}s. "
              "Review the wording and facts, then check the narration length.")
    if abs(board.word_count - word_budget(int(seconds))) > word_budget(int(seconds)) * 0.1:
        status += " The draft is outside the estimated word budget; a timing check will show whether it needs revision."
    return board.title, board.rows(), status, gr.Tabs(selected="edit"), seconds


def load_example():
    return EXAMPLE.title, EXAMPLE.rows(), "Example storyboard loaded. This is a prewritten demo; edit it or generate your own.", gr.Tabs(selected="edit"), 30


def timing_message(durations, seconds):
    summary = f"**Narration: {sum(durations):.1f} seconds · Target: {seconds} seconds.** "
    try:
        plan = plan_timing(durations, seconds)
    except DurationMismatch:
        action = "Expand" if sum(durations) < seconds else "Shorten"
        return summary + f"{action} the script to fit, edit it yourself, or choose another length."
    return summary + f"Ready to fit with gentle pacing adjustments. Expected export: {sum(plan.scene_seconds):.1f} seconds."


def check_narration(title, rows, voice, seconds, progress=gr.Progress()):
    job = None
    try:
        validate_target(seconds)
        board = from_editor(title, rows)
        progress(0, desc="Measuring narration with your selected voice")
        clean_old_exports()
        job = Path(tempfile.mkdtemp(prefix="video-", dir=OUTPUTS))
        audio = job / "narration.wav"
        durations = measure_narration([scene.narration for scene in board.scenes], voice, audio)
        try:
            plan_timing(durations, seconds)
            revision_button = gr.update(value="Script fits target", interactive=False)
        except DurationMismatch:
            action = "Expand" if sum(durations) < seconds else "Shorten"
            revision_button = gr.update(value=f"{action} script to fit", interactive=True)
        return timing_message(durations, seconds), gr.update(value=str(audio), visible=True), revision_button
    except Exception as exc:
        if job is not None:
            shutil.rmtree(job, ignore_errors=True)
        logger.exception("Narration check failed")
        raise gr.Error(str(exc) if isinstance(exc, ValueError) else "Narration could not be measured. Please try again.") from exc


def revision_key(board, voice, seconds):
    return board.model_dump_json(), voice, seconds


def suggest_revision(title, rows, voice, seconds, notes, progress=gr.Progress()):
    try:
        validate_target(seconds)
        board = from_editor(title, rows)
        progress(0, desc="Measuring your script before revising it")
        durations = measure_narration([scene.narration for scene in board.scenes], voice)
        progress(0.3, desc="Revising narration to fit · waiting for the script model")
        revised = revise_storyboard(board, seconds, sum(durations), notes)
        progress(0.8, desc="Measuring the suggested revision")
        revised_durations = measure_narration([scene.narration for scene in revised.scenes], voice)
        state = {"base": revision_key(board, voice, seconds), "rows": revised.rows()}
        review = "**Suggested revision — your current script is unchanged.** " + timing_message(revised_durations, seconds)
        return gr.update(visible=True), [[scene.heading, scene.narration] for scene in revised.scenes], review, state
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Script revision failed")
        raise gr.Error("The revision could not finish. Your script is unchanged. If your GPU allowance is exhausted, try later or edit it yourself.") from exc


def accept_revision(title, rows, voice, seconds, revision):
    board = from_editor(title, rows)
    if not revision or revision.get("base") != revision_key(board, voice, seconds):
        raise gr.Error("The script, voice or target changed after this revision was requested. Generate a fresh suggestion to keep your edits safe.")
    return revision["rows"], "Revision applied. Review your script before rendering.", None, gr.update(visible=False)


def make_video(title, rows, voice, aspect, uploads, ai_motion=False, target_seconds=30, progress=gr.Progress()):
    try:
        board = from_editor(title, rows)
        video, bundle, gallery, stats = render_storyboard(
            board, voice, aspect, uploads,
            progress=lambda value, description: progress(value, desc=description),
            ai_motion=ai_motion,
            target_seconds=target_seconds,
        )
    except ValidationError as exc:
        raise gr.Error(f"Check the storyboard: every scene needs a heading, narration, and image search. Keep it to {MAX_WORDS} spoken words or fewer.") from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Video render failed")
        if ai_motion:
            raise gr.Error("AI motion could not finish. The GPU may be busy or your daily allowance may be exhausted. Try later, or turn off AI motion to make a photo video.") from exc
        raise gr.Error("The video could not be completed. Please retry, or use your own images if image search is unavailable.") from exc
    info = (f"**Your video is ready.** {stats['duration_seconds']:.0f} seconds · "
            f"{target_seconds}-second target · "
            f"{len(board.scenes)} scenes · rendered in {stats['render_seconds']:.0f}s. "
            "Download the bundle for your MP4, captions, storyboard, and image credits.")
    if ai_motion:
        info += " **AI motion enabled:** each image was animated with LTX-Video and adjusted to fit its narration. Review the generated visuals before sharing."
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
                            seconds = gr.Radio(list(LENGTHS), value=30, label="Target length · seconds")
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
                    edit_seconds = gr.Radio(list(LENGTHS), value=30, label="Target length · seconds")
                length_status = gr.Markdown("Check the narration length to hear your script and see whether it fits.")
                with gr.Row():
                    check = gr.Button("Check narration length", variant="secondary")
                    revise = gr.Button("Revise script to fit", variant="secondary", interactive=False)
                length_feedback = gr.HTML("", elem_classes="job-feedback")
                revision_feedback = gr.HTML("", elem_classes="job-feedback")
                narration = gr.Audio(label="Narration preview · before pacing adjustments", type="filepath", interactive=False, visible=False)
                revision_state = gr.State(None)
                with gr.Column(visible=False) as revision_panel:
                    revision_status = gr.Markdown()
                    revision_rows = gr.Dataframe(headers=["Scene", "Suggested narration"],
                                                 datatype="str", type="array", interactive=False, wrap=True)
                    with gr.Row():
                        accept = gr.Button("Use this revision", variant="primary")
                        discard = gr.Button("Keep my current script", variant="secondary")
                gr.Markdown("Checking narration uses no GPU. A suggested revision uses the script model and stays separate until you accept it.")
                with gr.Accordion("Use your own images · optional", open=False):
                    uploads = gr.File(label="One image for each scene, in scene order", file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                    gr.Markdown("Leave this empty to search Wikimedia Commons. Upload images you have permission to use.")
                ai_motion = gr.Checkbox(value=False, label="Animate images with AI · experimental",
                                        interactive=MOTION_AVAILABLE,
                                        info="Generate moving footage from each image and its storyboard text. Slower; uses your free GPU allowance.")
                motion_info = gr.Markdown(
                    "LTX-Video creates about **4 seconds of motion per scene**, then playback is adjusted to fit the narration. "
                    "Clips are generated at 512 × 288 (or portrait) and exported at 720p. "
                    "Generated details can differ from the original photo. "
                    "By enabling AI motion, you agree to the [LTX model use terms](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt).",
                    visible=False)
                ai_motion.change(lambda enabled: gr.update(visible=enabled), ai_motion, motion_info,
                                 queue=False, api_name=False)
                if not MOTION_AVAILABLE:
                    gr.Markdown("AI motion is temporarily unavailable. You can still create photo videos.")
                render = gr.Button("Make my video  →", variant="primary", size="lg")
                render_feedback = gr.HTML("", elem_id="video-progress", elem_classes="job-feedback")
                gr.Markdown("Your video will fit the selected length with small pacing changes. If the script needs a substantial rewrite, we’ll flag it before rendering.")
            with gr.Tab("03 · Watch & download", id="watch"):
                result = gr.Markdown("Your finished video will appear here.")
                video = gr.Video(label="Your short video", interactive=False, height=520, buttons=["download"])
                bundle = gr.File(label="Download the full bundle · MP4 + captions + storyboard + credits", interactive=False)
                with gr.Accordion("Images used in this video", open=False):
                    gallery = gr.Gallery(label="Scene images", columns=4, height=230, interactive=False)
                gr.Markdown("Keep **credits.txt** with your video when sharing Commons images. Downloads are temporary; save anything you want to keep.")
        gr.HTML('<footer class="foot"><span>Made with open models. Finished by you.</span><span>Qwen · Kokoro · LTX-Video · Wikimedia Commons</span></footer>')
        writing = loading_event(write, write_feedback,
                                "Writing your storyboard… A GPU may need to become available first.")
        writing = writing.then(draft, [topic, notes, tone, seconds], [title, editor, status, tabs, edit_seconds],
                               concurrency_limit=1, concurrency_id="writer", api_name="draft",
                               show_progress="minimal", show_progress_on=[write_feedback])
        clear_loading(writing, write, write_feedback)
        example.click(load_example, outputs=[title, editor, status, tabs, edit_seconds], queue=False, api_name=False)
        # User-only input events avoid a feedback loop between the two controls.
        seconds.input(lambda value: value, seconds, edit_seconds, queue=False, api_name=False)
        edit_seconds.change(lambda value: value, edit_seconds, seconds, queue=False, api_name=False)
        checking = loading_event(check, length_feedback, "Checking narration length with your selected voice…")
        checking = checking.then(check_narration, [title, editor, voice, edit_seconds],
                                 [length_status, narration, revise], concurrency_limit=1,
                                 concurrency_id="renderer", api_name="check_narration",
                                 show_progress="minimal", show_progress_on=[length_feedback])
        clear_loading(checking, check, length_feedback)
        revising = loading_event(revise, revision_feedback, "Preparing a suggested revision… Your current script stays unchanged.")
        revising = revising.then(suggest_revision, [title, editor, voice, edit_seconds, notes],
                                 [revision_panel, revision_rows, revision_status, revision_state],
                                 concurrency_limit=1, concurrency_id="renderer", api_name="suggest_revision",
                                 show_progress="minimal", show_progress_on=[revision_feedback])
        clear_loading(revising, revise, revision_feedback)
        accept.click(accept_revision, [title, editor, voice, edit_seconds, revision_state],
                     [editor, status, revision_state, revision_panel], queue=False, api_name=False)
        discard.click(lambda: (None, gr.update(visible=False)), outputs=[revision_state, revision_panel],
                      queue=False, api_name=False)
        rendering = loading_event(render, render_feedback,
                                  "Making your video… Preparing your scenes. AI motion may also need to wait for a GPU.")
        rendering = rendering.then(make_video, [title, editor, voice, aspect, uploads, ai_motion, edit_seconds],
                                    [video, bundle, gallery, result, tabs], concurrency_limit=1,
                                    concurrency_id="renderer", api_name="render",
                                    show_progress="minimal", show_progress_on=[render_feedback])
        clear_loading(rendering, render, render_feedback)
        # An old video must never look like the output of newly edited settings.
        def stale():
            return None, None, [], "Your storyboard changed. Render again to see the updated video."
        for control in (title, editor, voice, aspect, uploads, ai_motion, edit_seconds):
            control.change(stale, outputs=[video, bundle, gallery, result], queue=False, api_name=False)
        for control in (title, editor, voice, edit_seconds):
            control.change(lambda: ("Narration or target changed. Check the length again before rendering.",
                                     gr.update(value=None, visible=False), gr.update(value="Revise script to fit", interactive=False),
                                     None, gr.update(visible=False)),
                           outputs=[length_status, narration, revise, revision_state, revision_panel],
                           queue=False, api_name=False)
    return demo


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    theme = gr.themes.Base(primary_hue="lime", secondary_hue="slate", neutral_hue="stone",
                           font=["Inter", "Arial", "sans-serif"], text_size="lg")
    build_app().queue(max_size=12).launch(
        server_name="0.0.0.0" if IS_SPACE else "127.0.0.1",
        server_port=int(os.getenv("PORT", "7865" if not IS_SPACE else "7860")),
        theme=theme, css_paths=ROOT / "assets/style.css", max_file_size="15mb",
        js=(ROOT / "assets/resize-embed.js").read_text(),
        allowed_paths=[str(OUTPUTS)], footer_links=[], show_error=False,
    )

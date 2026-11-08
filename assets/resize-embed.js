() => {
    if (window.parent === window) return;
    const app = document.querySelector(".gradio-container");
    if (!app) return;

    // Spaces disables scrolling inside the iframe. Its height must also follow
    // tab switches, narration previews, revisions and expanded image controls.
    let scheduled = false;
    const resize = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            window.parentIFrame?.size();
        });
    };
    new ResizeObserver(resize).observe(app);
    window.addEventListener("pageshow", resize);
    resize();
}

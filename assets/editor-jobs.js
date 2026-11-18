() => {
    if (window.storycutJobs) return;
    // This state belongs to one browser tab. Model and file outputs arrive in
    // hidden Gradio components. Only a current job can publish them to the editor.
    const versions = {draft: 0, check_narration: 0, suggest_revision: 0, accept_revision: 0, render: 0};
    const started = new Map();
    const update = (props = {}) => ({__type__: "update", ...props});
    let canRevise = false;

    function invalidate(...names) {
        for (const name of names) versions[name]++;
    }

    function clearVideo() {
        return [null, null, [], "Your storyboard changed. Render again to see the updated video."];
    }

    function clearTiming() {
        canRevise = false;
        return [
            "Narration or target changed. Check the length again before rendering.",
            update({value: null, visible: false}),
            update({value: "Revise script to fit", interactive: false}),
            update({visible: false}),
        ];
    }

    function scriptChanged() {
        invalidate(...Object.keys(versions));
        return [...clearVideo(), ...clearTiming()];
    }

    function mediaChanged() {
        invalidate("draft", "render");
        return clearVideo();
    }

    function project(title, rows, message, seconds) {
        return [title, rows, message, update({selected: "edit"}), seconds,
            null, false, ...scriptChanged()];
    }

    const publish = {
        draft: project,
        check_narration(message, audio, action) {
            canRevise = action !== "Script fits target";
            return [message, update({value: audio, visible: true}),
                update({value: action, interactive: canRevise && !started.has("suggest_revision")})];
        },
        suggest_revision(rows, message) {
            return [update({visible: true}), rows, message];
        },
        accept_revision(rows, message) {
            return [rows, message, ...scriptChanged()];
        },
        render(video, bundle, gallery, message) {
            return [video, bundle, gallery, message, update({selected: "watch"})];
        },
    };

    function release(name) {
        started.delete(name);
        return [update({interactive: name !== "suggest_revision" || canRevise}), ""];
    }

    window.storycutJobs = {
        invalidate, scriptChanged, mediaChanged, project, release,
        notesChanged() {
            invalidate("draft", "suggest_revision", "accept_revision");
            return [update({visible: false})];
        },
        discard() {
            invalidate("suggest_revision", "accept_revision");
            return [update({visible: false})];
        },
        begin(name, markup) {
            if (!started.has(name)) {
                if (name === "draft") invalidate(...Object.keys(versions));
                started.set(name, versions[name]);
            }
            const controls = [update({interactive: false}), markup];
            if (name === "suggest_revision") {
                invalidate("accept_revision");
                controls.push(update({visible: false}));
            }
            return controls;
        },
        finish(name, outputCount, ...values) {
            const current = started.get(name) === versions[name];
            const outputs = current ? publish[name](...values) : Array.from({length: outputCount}, () => update());
            return [...outputs, ...release(name)];
        },
    };

    // Text components can commit on blur. Invalidate immediately, even for an
    // uncommitted title or an open dataframe cell, before a job can publish.
    document.addEventListener("input", (event) => {
        const field = event.target;
        if (field.closest?.("#story-title, #storyboard-editor")) {
            invalidate(...Object.keys(versions));
            canRevise = false;
        } else if (field.closest?.("#source-notes")) {
            invalidate("draft", "suggest_revision", "accept_revision");
        } else if (field.closest?.("#story-topic")) {
            invalidate("draft");
        }
    }, true);
}

"""Create a local, paired video review and sampled-frame contact sheets."""

import html
import json
from pathlib import Path
import statistics

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".cache/motion-benchmark"


def main():
    cases = json.loads((DATA / "cases.json").read_text())
    records = {model: [json.loads(p.read_text()) for p in sorted((DATA / model).glob("*.json"))]
               for model in ("ltx", "fastwan")}
    summary = {}
    for model, rows in records.items():
        if not rows:
            continue
        summary[model] = {"runs": len(rows), "median": {
            key: statistics.median(r[key] for r in rows)
            for key in ("gpu_body_seconds", "server_seconds", "client_seconds", "peak_allocated_gb")}}
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2))
    sections = []
    for case in cases:
        portrait = case["portrait"]
        tile_width, tile_height = (135, 240) if portrait else (240, 135)
        sheet = Image.new("RGB", (9 * tile_width, 4 * (tile_height + 30)), "#faf9f5")
        draw = ImageDraw.Draw(sheet)
        for seed_index, seed in enumerate((42, 314)):
            videos = []
            for model_index, model in enumerate(("ltx", "fastwan")):
                path = DATA / model / f"{case['name']}-{seed}.mp4"
                record = path.with_suffix(".json")
                if not path.exists() or not record.exists():
                    continue
                metrics = json.loads(record.read_text())
                reader = imageio_ffmpeg.read_frames(str(path))
                meta = next(reader)
                width, height = meta["size"]
                count = 0
                y = (seed_index * 2 + model_index) * (tile_height + 30)
                for index, frame in enumerate(reader):
                    count += 1
                    if index % 12 == 0:
                        picture = Image.fromarray(np.frombuffer(frame, np.uint8).reshape(height, width, 3))
                        picture = picture.resize((tile_width, tile_height))
                        x = index // 12 * tile_width
                        sheet.paste(picture, (x, y + 30))
                        draw.text((x + 3, y + 8), f"{model} {seed} {index/24:.1f}s", fill="black")
                assert count == 97 and meta["fps"] == 24
                videos.append(f'<figure><figcaption>{model} · GPU {metrics["gpu_body_seconds"]:.2f}s</figcaption>'
                              f'<video controls muted loop playsinline preload="metadata" '
                              f'src="{model}/{case["name"]}-{seed}.mp4"></video></figure>')
            sections.append(f'<section><h2>{html.escape(case["name"])} · seed {seed}</h2>'
                            f'<button onclick="playPair(this)">Play both from start</button>'
                            f'<div class="pair">{"".join(videos)}</div></section>')
        sheet.save(DATA / f"{case['name']}-comparison.jpg")
    credit_items = [f'<li>{html.escape(c["case"])}: '
                    f'<a href="{html.escape(c["source_url"], quote=True)}">{html.escape(c["title"])}</a> '
                    f'by {html.escape(c["creator"])} · {html.escape(c["license"])}</li>'
                    for c in json.loads((DATA / "credits.json").read_text())]
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Storycut motion comparison</title><style>
body{font:18px/1.5 system-ui;background:#f7f6f0;color:#20372d;max-width:1100px;margin:32px auto;padding:20px}
.pair{display:flex;gap:24px;flex-wrap:wrap}figure{flex:1;min-width:260px;margin:12px 0}
video{width:100%;max-height:500px;background:#172820}section{padding:20px 0;border-top:1px solid #cbd4c0}
button{padding:10px 20px;font:inherit;color:white;background:#203d30;border:0;border-radius:8px;cursor:pointer}
pre{white-space:pre-wrap;font-size:14px}a{color:inherit}
</style><h1>Storycut: LTX vs FastWan</h1>
<p>Same images, prompts, seeds, four-second clips and 48GB ZeroGPU allocation. These are generated visual interpretations, not original footage. GPU time excludes queueing and encoding.</p>
<p><strong>Decision: keep LTX as Storycut's default.</strong> Across eight matched pairs, the tested FastWan configuration was slower and did not deliver a consistent visual improvement. This comparison does not cover FastVideo's native custom kernels. See <a href="../../benchmarks/motion/README.md">the written report</a> for method and limitations.</p>
'''
    page += "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>"
    page += "".join(sections)
    probe_path = DATA / "fastwan-untiled/walker-42.json"
    if probe_path.exists() and probe_path.with_suffix(".mp4").exists():
        probe = json.loads(probe_path.read_text())
        page += (f'<section><h2>Additional untiled FastWan probe</h2>'
                 f'<p>One completed first-call sample: {probe["gpu_body_seconds"]:.2f}s GPU-function time, '
                 f'{probe["peak_allocated_gb"]:.2f} GB peak allocated memory. '
                 'This is not an eight-run median. The next optional request was blocked by the free quota reservation limit.</p>'
                 '<video controls muted loop playsinline preload="metadata" '
                 'src="fastwan-untiled/walker-42.mp4"></video></section>')
    page += "<h2>Source-image credits</h2><ul>" + "".join(credit_items) + "</ul>"
    page += '''<script>function playPair(button){button.parentElement.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});}</script></html>'''
    (DATA / "report.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

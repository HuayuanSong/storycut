# Storycut motion-model comparison — 6 November 2026

**Decision: keep LTX-Video 2B 0.9.8 distilled as the live default.** FastWan did not meet the requirement to improve quality without increasing runtime in this deployment configuration. No production application files or deployed model were changed by this experiment.

## Results

Eight matched pairs: four source photographs × two seeds. Each generated clip contains 97 frames at 24 fps, at 288×512 portrait or 512×288 landscape. Both used the same default ZeroGPU allocation: RTX PRO 6000 Blackwell MIG 2g.48gb. The existing Qwen storyboard model remained resident in both trials, matching the application's memory requirements.

| Measurement | LTX 2B distilled, 8 steps | FastWan 5B, 3-step DMD |
| --- | ---: | ---: |
| Median GPU-function runtime per clip | 1.29 s | 10.38 s |
| Median server time including GPU allocation/queue and MP4 encoding | 1.82 s | 11.22 s |
| Median client time including upload, queue and download | 3.75 s | 13.21 s |
| Median peak allocated GPU memory, including Qwen | 25.47 GB | 34.03 GB |
| Completed clips | 8 | 8 |

FastWan used approximately **8.0×** the measured GPU-function time. Removing spatial VAE tiling improved one additional FastWan probe to **7.41 s**, with **35.89 GB** peak allocated memory. That is a single first-call measurement, not an eight-run median. Its three denoising steps finished at 2.38 s; decoding and frame conversion occupied the remaining roughly 5.04 s.

The next untiled probe was rejected before generation because Hugging Face requested a 90-second reservation with 86 seconds left. Consequently, the untiled configuration has only one completed sample. The matched eight-pair comparison was already complete. An earlier missing `ftfy` dependency was fixed before collecting FastWan results; that failed request is excluded.

## Visual findings

Inspection used contact sheets sampled every half second, including the final frame. The local comparison page also provides all full clips for playback. These are qualitative observations, not a blinded human preference study or an automatic quality score.

- **Walking person:** FastWan maintained a more coherent walking direction. Both models invented details as more of the person came into view; FastWan also introduced footwear despite the source showing a bare foot. LTX had less plausible pose changes in one seed.
- **Foliage:** LTX maintained more consistent colouring. FastWan introduced stronger lighting/colour changes across the sampled frames.
- **Cat:** Both generated plausible head movement. FastWan changed contrast and appearance more strongly, without a consistent improvement in source-image fidelity.
- **Coffee:** Both preserved the main scene. One FastWan sample became noticeably defocused toward the end; LTX remained more consistent across the two seeds.

These examples do not establish a universal quality ranking. They do establish that this FastWan configuration is slower and does not provide a consistent enough visual improvement to replace the current default under the user's constraint.

## Method and scope

- Inputs, image crops, prompts, seeds (42 and 314), dimensions, frame counts, GPU type and PyTorch version were verified identical for every pair. Input SHA-256 hashes and prompts are retained in [the raw results](results/2026-11-06.json).
- GPU-function time is measured with CUDA synchronization around the pipeline call. It includes text/image encoding, denoising, decoding and frame conversion. It excludes allocation/setup before function entry, queueing and MP4 encoding. It is **not an exact measurement of Hugging Face's billed quota**.
- Startup model downloads are excluded; the first inference is included in each eight-run median. Each model was evaluated in its own fresh private preview instance. Queueing and client transfer times vary, so the GPU-function measurement is the main comparison.
- LTX runs the application's existing BF16 pipeline. FastWan uses BF16 transformer/text encoding, the recommended FP32 VAE, full attention and a small DMD scheduler adapter with Diffusers' image-conditioning path. The main FastWan trial uses spatial VAE tiling; the additional probe disables it.
- The DMD adapter follows the authors' three-step ladder `[1000, 757, 522]`, clean-prediction/re-noising equations and shift-8 training-noise lookup. Its numerical updates were checked for exact equality with the pinned upstream clean-prediction helper and re-noising equations on small FP32 and BF16 tensors. This is not an end-to-end parity claim for the native FastVideo runtime.
- This experiment does **not** benchmark FastVideo's native custom kernels, sparse-attention variants, other quantizations, native 720p generation, LightX2V or LTX-2.5. Those remain separate candidates, not disproven alternatives.
- The public Storycut Space remained on commit `c4d39d52b3ac626378f1ddd6d8f6e7a6ad24430e` throughout. The temporary preview is deleted after the experiment.

## Reproducing and reviewing locally

The fixed inputs, downloaded clips and browser comparison are retained in `.cache/motion-benchmark/`. Open `.cache/motion-benchmark/report.html` to play matched clips. Image attribution is included there and in the raw results; generated clips are cropped, AI-modified interpretations of those photographs. Binary media is excluded from Git.

`space_app.py` is an isolated Gradio benchmark entrypoint. A private ZeroGPU Space needs the existing project requirements plus `requirements-extra.txt`, the `storycut` modules, and this directory. Set its `app_file` to `benchmarks/motion/space_app.py`, use the same Python/Gradio versions as the main project, and set `STORYCUT_BENCH_MODEL` to `ltx` or `fastwan`. For the original FastWan trial set `STORYCUT_BENCH_TILING=1`; for the untiled probe use `0`. Only one model is loaded per preview to respect the free allocation and avoid filling its disk with both checkpoints.

With the prepared input manifest in `.cache/motion-benchmark/cases.json`:

```sh
.venv/bin/python benchmarks/motion/run.py YOUR_PRIVATE_SPACE ltx
.venv/bin/python benchmarks/motion/run.py YOUR_PRIVATE_SPACE fastwan
.venv/bin/python benchmarks/motion/run.py YOUR_PRIVATE_SPACE fastwan --label fastwan-untiled --seeds 42
.venv/bin/python benchmarks/motion/summarize.py
```

The runner uses the existing local Hugging Face credential store. It never puts tokens in files or reports, skips already completed cases, and stops on an error. The comparison consumes the signed-in user's shared free GPU quota; allow for that before repeating it.

## Pinned sources

- [LTX-Video checkpoint repository](https://huggingface.co/Lightricks/LTX-Video/tree/8984fa25007f376c1a299016d0957a37a2f797bb): `ltxv-2b-0.9.8-distilled.safetensors`.
- [FastWan checkpoint repository](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers/tree/3e187042a324f6f5fb68fd22110a78725253de8f).
- [FastVideo's DMD update implementation](https://github.com/hao-ai-lab/FastVideo/blob/a01c50a8063386895c6d8895a1e2509d11d55284/fastvideo/pipelines/basic/wan/stages/dmd.py), [clean prediction](https://github.com/hao-ai-lab/FastVideo/blob/a01c50a8063386895c6d8895a1e2509d11d55284/fastvideo/models/utils.py), and [training-noise shift](https://github.com/hao-ai-lab/FastVideo/blob/a01c50a8063386895c6d8895a1e2509d11d55284/fastvideo/models/wan/definition.py).

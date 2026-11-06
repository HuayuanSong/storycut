"""Run fixed inputs against a private benchmark Space using the signed-in account."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from gradio_client import Client, handle_file
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".cache/motion-benchmark"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("backend", choices=["ltx", "fastwan"])
    parser.add_argument("--label", help="Separate output folder for a configuration trial")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314])
    args = parser.parse_args()
    api = HfApi(token=True)
    previous = None
    for _ in range(90):
        runtime = api.get_space_runtime(args.repo)
        if runtime.stage != previous:
            print(runtime.stage, flush=True)
            previous = runtime.stage
        if runtime.stage == "RUNNING":
            break
        if runtime.stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
            raise RuntimeError(runtime.raw.get("errorMessage", runtime.stage))
        time.sleep(10)
    else:
        raise RuntimeError("Preview did not become ready")
    client = Client(args.repo, token=get_token(), verbose=False, analytics_enabled=False)
    output = DATA / (args.label or args.backend)
    output.mkdir(exist_ok=True)
    cases = json.loads((DATA / "cases.json").read_text())
    for seed in args.seeds:
        for case in cases:
            name = f"{case['name']}-{seed}"
            record = output / f"{name}.json"
            if record.exists():
                continue
            path = DATA / case["image"]
            started = time.perf_counter()
            print("Starting", args.backend, name, flush=True)
            result = client.predict(handle_file(str(path)), case["prompt"], seed,
                                    case["portrait"], api_name="/benchmark")
            wall = time.perf_counter() - started
            video, metrics = result
            assert metrics["backend"] == args.backend
            metrics.update(case=case["name"], client_seconds=wall,
                           input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           prompt=case["prompt"], run_order=len(list(output.glob("*.json"))))
            shutil.copyfile(video, output / f"{name}.mp4")
            record.write_text(json.dumps(metrics, indent=2))
            print(json.dumps({k: metrics[k] for k in (
                "case", "seed", "gpu_body_seconds", "server_seconds", "client_seconds",
                "peak_allocated_gb", "device")}), flush=True)


if __name__ == "__main__":
    main()

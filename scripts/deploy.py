"""Publish only this app to free ZeroGPU. Never select paid fallback hardware."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cache/deployment.json"
UPLOADS = ["app.py", "storycut/*.py", "assets/*", "README.md", "LICENSE", "requirements.txt", "packages.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="storycut")
    parser.add_argument("--check", action="store_true", help="Validate deployment files without using an account.")
    args = parser.parse_args()
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.name):
        parser.error("Use a lowercase Space name with letters, numbers, and hyphens.")
    for name in ("app.py", "requirements.txt", "README.md", "assets/style.css", "assets/Figtree.ttf"):
        if not (ROOT / name).is_file():
            raise SystemExit(f"Required deployment file is missing: {name}")
    if args.check:
        print("Deployment files present. Target: Gradio / ZeroGPU (zero-a10g). No paid fallback.")
        print("Uploaded paths:", ", ".join(UPLOADS))
        return
    api = HfApi(token=True)
    try:
        account = api.whoami()["name"]
    except Exception:
        raise SystemExit("Sign in first using .venv/bin/hf auth login. Your token stays in Hugging Face's local credential store.") from None
    repo = f"{account}/{args.name}"
    recorded = json.loads(STATE.read_text()) if STATE.exists() else {}
    if recorded.get("repo_id") != repo:
        try:
            api.repo_info(repo, repo_type="space")
        except RepositoryNotFoundError:
            pass
        else:
            raise SystemExit(f"{repo} already exists and was not created by this checkout. Choose another --name.")
        try:
            api.create_repo(repo_id=repo, repo_type="space", space_sdk="gradio",
                            space_hardware=SpaceHardware.ZERO_A10G, private=False)
        except Exception as exc:
            raise SystemExit("Hugging Face did not create the ZeroGPU Space. Check your account's ZeroGPU eligibility and token permissions. No paid hardware was requested.") from exc
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({"repo_id": repo}, indent=2))
    runtime = api.get_space_runtime(repo)
    hardware = runtime.requested_hardware or runtime.hardware
    if hardware != SpaceHardware.ZERO_A10G:
        raise SystemExit(f"The Space is configured as {hardware}, not ZeroGPU. Nothing was uploaded.")
    result = api.upload_folder(repo_id=repo, repo_type="space", folder_path=ROOT,
                               allow_patterns=UPLOADS, commit_message="Deploy Storycut video studio")
    print(f"Uploaded commit {result.oid}")
    print(f"Build status: https://huggingface.co/spaces/{repo}")
    print("The first build downloads the models. Check that the Space reaches RUNNING before sharing it.")


if __name__ == "__main__":
    main()

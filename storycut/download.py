"""Small, fixed-origin model asset cache."""

from pathlib import Path
from threading import Lock
import hashlib
import os

import httpx

_lock = Lock()


def cached_release(filename: str) -> Path:
    if filename != "voices-v1.0.bin":
        raise ValueError("Unknown model asset.")
    root = Path(os.getenv("STORYCUT_CACHE", Path.home() / ".cache/storycut"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    with _lock:
        if not path.exists():
            url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/{filename}"
            temporary = path.with_suffix(".part")
            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=90) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        total = 0
                        for block in response.iter_bytes():
                            total += len(block)
                            if total > 40_000_000:
                                raise ValueError("Unexpected model asset size.")
                            output.write(block)
                expected = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                    raise ValueError("The downloaded voice bank failed its integrity check.")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return path

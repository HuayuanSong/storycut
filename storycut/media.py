"""Keyless Commons image search with explicit licensing and bounded downloads."""

import html
import re
from functools import lru_cache
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

HEADERS = {"User-Agent": "Storycut/0.1 (https://github.com/huayuansong; open-source video editor)"}
HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}
MAX_BYTES = 15_000_000
Image.MAX_IMAGE_PIXELS = 32_000_000


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", value))).strip()


@dataclass
class Visual:
    path: Path
    title: str
    creator: str
    license: str
    license_url: str
    source_url: str

    def credit(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "path"}


def permitted_license(name: str, url: str) -> bool:
    # Keep the initial export simple: public domain, CC0, or attribution-only.
    parsed = urlparse(url)
    if name.lower() in {"public domain", "cc0", "cc0 1.0"}:
        return True
    return parsed.hostname == "creativecommons.org" and parsed.path.startswith("/licenses/by/")


def save_image(data: bytes, destination: Path) -> None:
    if len(data) > MAX_BYTES:
        raise ValueError("Each image must be smaller than 15 MB.")
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Use a JPEG, PNG, or WebP image.")
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("Use an image below 32 megapixels.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1920, 1920))
        image.save(destination, "JPEG", quality=92)


def upload_visual(source: str, destination: Path) -> Visual:
    path = Path(source)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Each image must be smaller than 15 MB.")
    save_image(path.read_bytes(), destination)
    return Visual(destination, path.name, "User upload", "Provided by user", "", "")


@lru_cache(maxsize=128)
def search_commons(query: str) -> list[dict]:
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 40,
              "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": 1280,
              "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|ObjectName"}
    response = httpx.get("https://commons.wikimedia.org/w/api.php", params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {}).values()
    return sorted(pages, key=lambda item: item.get("index", 0))


def commons_visual(query: str, destination: Path, used: set[str]) -> Visual:
    query = re.sub(r"[^\w\s-]", " ", query)[:100].strip()
    # A broader noun search helps when a generated shot description is too specific.
    queries = list(dict.fromkeys([query, " ".join(query.split()[:1])]))
    with httpx.Client(headers=HEADERS, timeout=20, follow_redirects=False) as client:
        pages = (page for search in queries for page in search_commons(search))
        attempts = 0
        for page in pages:
            info = next(iter(page.get("imageinfo", [])), {})
            meta = info.get("extmetadata", {})
            value = lambda key: clean_text(str(meta.get(key, {}).get("value", "")))
            license_name, license_url = value("LicenseShortName"), value("LicenseUrl")
            url = info.get("thumburl", "")
            if (not permitted_license(license_name, license_url)
                    or urlparse(url).hostname not in HOSTS or urlparse(url).scheme != "https"
                    or info.get("width", 0) < 600 or info.get("height", 0) < 400
                    or page["title"] in used):
                continue
            try:
                attempts += 1
                if attempts > 4:
                    break
                with client.stream("GET", url) as image_response:
                    image_response.raise_for_status()
                    data = bytearray()
                    for block in image_response.iter_bytes():
                        data.extend(block)
                        if len(data) > MAX_BYTES:
                            raise ValueError("Image is too large.")
                save_image(bytes(data), destination)
            except (httpx.HTTPError, OSError, ValueError, Image.DecompressionBombError):
                continue
            used.add(page["title"])
            return Visual(destination, value("ObjectName") or page["title"],
                          value("Artist") or "See source page", license_name, license_url,
                          info.get("descriptionurl", ""))
    raise ValueError(f'No suitable reusable image found for “{query}”. Try a broader image search or upload your own pictures.')

"""
Read apify_results.json, group by ownerUsername, write:

  data/influencers/<username>/posts.json
  data/influencers/_index.json
  data/influencers/<username>/images/   (empty; for downloaded files)

Imágenes: el export completo (p. ej. dataset_* (1).json) incluye displayUrl
y, en carruseles, a veces "images" con varias URLs. Con --download se guardan
junto a cada username. Usa --include-videos para bajar también el thumbnail
de posts type=Video (por defecto se omiten, alineado con "solo fotos").

Geo (país/ciudad): edita data/influencers/influencers_geo.json a mano; el
pipeline de matching leerá ese JSON además de los posts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from ingest.paths import (
    DATA_INFLUENCERS_DIR,
    INFLUENCERS_GEO_JSON,
    REPO_ROOT,
    get_apify_export_path,
)

# Campos de imagen que puedes añadir al JSON tras otro run de Apify
IMAGE_URL_KEYS = (
    "displayUrl",
    "imageUrl",
    "thumbnailUrl",
    "image",
    "thumbnail",
)


def _shortcode_from_post_url(url: str) -> str:
    m = re.search(r"instagram\.com/p/([^/?#]+)", url, re.I)
    return m.group(1) if m else "unknown"


def _first_image_url(post: dict[str, Any]) -> str | None:
    for k in IMAGE_URL_KEYS:
        v = post.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    images = post.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str) and first.startswith("http"):
            return first
        if isinstance(first, dict):
            for k in IMAGE_URL_KEYS:
                u = first.get(k)
                if isinstance(u, str) and u.startswith("http"):
                    return u
    return None


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ScuffersIngest/1.0; +https://scuffers.com)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"  skip download {dest.name}: {e}", file=sys.stderr)
        return False
    return True


def run(
    *, apify_path: Path, download: bool, skip_videos: bool
) -> None:
    raw = apify_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("apify file must be a JSON array of posts")

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        u = (item.get("ownerUsername") or "unknown").strip() or "unknown"
        u = u.lower()
        by_user[u].append(item)

    index: list[dict[str, Any]] = []
    root = DATA_INFLUENCERS_DIR
    root.mkdir(parents=True, exist_ok=True)

    for username in sorted(by_user):
        udir = root / username
        (udir / "images").mkdir(parents=True, exist_ok=True)
        posts = by_user[username]
        (udir / "posts.json").write_text(
            json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_dl = 0
        if download:
            for i, post in enumerate(posts, start=1):
                if skip_videos and (post.get("type") or "").lower() == "video":
                    continue
                img_url = _first_image_url(post)
                if not img_url:
                    continue
                sc = (
                    str(post.get("shortCode") or "").strip()
                    or _shortcode_from_post_url(str(post.get("url", "")))
                )
                ext = ".jpg"
                if "png" in img_url.lower()[:120]:
                    ext = ".png"
                dest = udir / "images" / f"{i:02d}_{sc}{ext}"
                if _download_file(img_url, dest):
                    n_dl += 1
        rel = f"influencers/{username}/posts.json"
        index.append(
            {
                "username": username,
                "post_count": len(posts),
                "posts_json": rel.replace("\\", "/"),
            }
        )
        print(f"{username}: {len(posts)} posts" + (f", {n_dl} images" if download else ""))

    try:
        rel_src = str(apify_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel_src = str(apify_path)
    (root / "_index.json").write_text(
        json.dumps(
            {
                "source_apify": rel_src.replace("\\", "/"),
                "influencers": index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not INFLUENCERS_GEO_JSON.exists():
        INFLUENCERS_GEO_JSON.parent.mkdir(parents=True, exist_ok=True)
        template = {u: {"country": None, "city": None} for u in by_user}
        INFLUENCERS_GEO_JSON.write_text(
            json.dumps(
                {"_edit": "Rellena country/city por username (ISO country opcional).", **template},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Created {INFLUENCERS_GEO_JSON} (edit country/city by hand if needed).")


def main() -> int:
    p = argparse.ArgumentParser(description="Split Apify array by ownerUsername")
    p.add_argument(
        "--apify",
        type=Path,
        default=None,
        help="Path to Apify export (default: latest mtime: dataset_*.json or apify_results.json)",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download images if image URL fields are present in JSON",
    )
    p.add_argument(
        "--include-videos",
        action="store_true",
        help="With --download, also download cover for type=Video (default: skip videos)",
    )
    args = p.parse_args()
    apify = args.apify or get_apify_export_path()
    if not apify.is_file():
        print(
            f"Not found: {apify} (put apify_results.json or dataset_instagram-post-scraper_*.json in repo root)",
            file=sys.stderr,
        )
        return 1
    print(f"Using: {apify.name}")
    run(
        apify_path=apify,
        download=args.download,
        skip_videos=not args.include_videos,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

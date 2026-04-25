"""Project paths (repo root, Apify export, etc.)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Default filename when you paste from clipboard
APIFY_POSTS_JSON = REPO_ROOT / "apify_results.json"
# Offline dataset: one folder per Instagram username
DATA_INFLUENCERS_DIR = REPO_ROOT / "data" / "influencers"
# Optional: fill country / city by hand (see split_apify_by_influencer output)
INFLUENCERS_GEO_JSON = REPO_ROOT / "data" / "influencers" / "influencers_geo.json"


def get_apify_export_path() -> Path:
    """
    Apify *Dataset* download is often named e.g. dataset_instagram-post-scraper_DATE.json
    (or the same with " (1)" if the browser saved a duplicate).
    Pick the most recently modified among those and apify_results.json, so the latest
    export wins without hardcoding a filename.
    """
    cands: list[Path] = list(
        REPO_ROOT.glob("dataset_instagram-post-scraper_*.json")
    ) + [APIFY_POSTS_JSON]
    existing = [p for p in cands if p.is_file()]
    if not existing:
        return APIFY_POSTS_JSON
    return max(existing, key=lambda p: p.stat().st_mtime)


__all__ = [
    "REPO_ROOT",
    "APIFY_POSTS_JSON",
    "get_apify_export_path",
    "DATA_INFLUENCERS_DIR",
    "INFLUENCERS_GEO_JSON",
]
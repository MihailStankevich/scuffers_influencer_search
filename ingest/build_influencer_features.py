"""
Build aggregated influencer features from an Apify Instagram export.

Filters strictly to usernames present in data/influencers/influencers_geo.json.
Outputs:
  - data/influencers/influencer_features.json
  - data/influencers/influencer_features.db (SQLite table: influencer_features)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ingest.paths import DATA_INFLUENCERS_DIR, INFLUENCERS_GEO_JSON, get_apify_export_path

OUT_JSON = DATA_INFLUENCERS_DIR / "influencer_features.json"
OUT_DB = DATA_INFLUENCERS_DIR / "influencer_features.db"

HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def _normalize_username(value: str | None) -> str:
    return (value or "").strip().lower()


def _extract_hashtags(post: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    raw = post.get("hashtags")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip().lower())
    if not tags:
        caption = post.get("caption") or ""
        if isinstance(caption, str):
            tags = [m.group(1).lower() for m in HASHTAG_RE.finditer(caption)]
    return tags


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _build_features(apify_items: list[dict[str, Any]], geo: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {_normalize_username(u) for u in geo.keys()}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in apify_items:
        username = _normalize_username(post.get("ownerUsername"))
        if username in allowed:
            grouped[username].append(post)

    rows: list[dict[str, Any]] = []
    for username in sorted(allowed):
        posts = grouped.get(username, [])
        likes_known: list[float] = []
        comments: list[float] = []
        hashtag_counter: Counter[str] = Counter()

        for p in posts:
            likes = p.get("likesCount")
            if isinstance(likes, (int, float)) and likes >= 0:
                likes_known.append(float(likes))

            c = p.get("commentsCount")
            if isinstance(c, (int, float)) and c >= 0:
                comments.append(float(c))

            hashtag_counter.update(_extract_hashtags(p))

        geo_data = geo.get(username, {}) if isinstance(geo.get(username), dict) else {}
        top_hashtags = [
            {"tag": tag, "count": count}
            for tag, count in hashtag_counter.most_common(20)
        ]

        rows.append(
            {
                "username": username,
                "country": geo_data.get("country"),
                "city": geo_data.get("city"),
                "post_count": len(posts),
                "likes_known_count": len(likes_known),
                "avg_likes": _avg(likes_known),
                "avg_comments": _avg(comments),
                "top_hashtags": top_hashtags,
            }
        )
    return rows


def _write_json(rows: list[dict[str, Any]], source_file: Path) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_apify": source_file.name,
        "filtered_by_geo_file": str(INFLUENCERS_GEO_JSON.name),
        "count_influencers": len(rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_sqlite(rows: list[dict[str, Any]]) -> None:
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUT_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS influencer_features (
              username TEXT PRIMARY KEY,
              country TEXT,
              city TEXT,
              post_count INTEGER NOT NULL,
              likes_known_count INTEGER NOT NULL,
              avg_likes REAL NOT NULL,
              avg_comments REAL NOT NULL,
              top_hashtags_json TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM influencer_features")
        conn.executemany(
            """
            INSERT INTO influencer_features (
              username, country, city, post_count, likes_known_count,
              avg_likes, avg_comments, top_hashtags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["username"],
                    r["country"],
                    r["city"],
                    r["post_count"],
                    r["likes_known_count"],
                    r["avg_likes"],
                    r["avg_comments"],
                    json.dumps(r["top_hashtags"], ensure_ascii=False),
                )
                for r in rows
            ],
        )
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build influencer_features from Apify export")
    parser.add_argument(
        "--apify",
        type=Path,
        default=None,
        help="Path to Apify JSON export. Default: latest dataset/apify_results file",
    )
    args = parser.parse_args()

    source = args.apify or get_apify_export_path()
    if not source.is_file():
        raise SystemExit(f"Apify export not found: {source}")
    if not INFLUENCERS_GEO_JSON.is_file():
        raise SystemExit(f"Geo file not found: {INFLUENCERS_GEO_JSON}")

    apify_items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(apify_items, list):
        raise SystemExit("Apify export must be a JSON array")
    geo = json.loads(INFLUENCERS_GEO_JSON.read_text(encoding="utf-8"))
    if not isinstance(geo, dict):
        raise SystemExit("influencers_geo.json must be a JSON object")

    rows = _build_features(apify_items, geo)
    _write_json(rows, source)
    _write_sqlite(rows)

    print(f"Using source: {source.name}")
    print(f"Influencers (geo filter): {len(rows)}")
    print(f"Wrote JSON: {OUT_JSON}")
    print(f"Wrote SQLite: {OUT_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

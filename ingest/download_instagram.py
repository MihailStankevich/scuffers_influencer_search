"""
Download the latest non-video photo posts for an Instagram profile into a
structured folder tree, plus a JSON manifest (lightweight "database" row).

Public profiles work without login until rate-limited; for heavy use, pass a
session via env INSTALOADER_SESSIONFILE or login interactively once.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import os
from typing import Any, Iterator, List, Optional

import instaloader

# Reels and IGTV are videos; we skip any post that is video-only in Instaloader's view.
# Carousels with mixed media: we keep only image pages (and skip video slides).


def _normalize_username(handle: str) -> str:
    h = handle.strip()
    if h.startswith("@"):
        h = h[1:]
    if not h or not re.match(r"^[A-Za-z0-9._]+$", h):
        raise ValueError(f"Invalid Instagram username: {handle!r}")
    return h.lower()


def _iter_photo_posts(
    profile: instaloader.Profile, limit: int
) -> Iterator[instaloader.Post]:
    """
    Yield up to `limit` posts that contribute at least one still image
    (single photo or carousel image slides; not video/reel-only posts).
    """
    n = 0
    for post in profile.get_posts():
        if n >= limit:
            break
        if post.is_video and post.typename != "GraphSidecar":
            # Pure video (feed video / reel as video post, etc.)
            continue
        if post.typename == "GraphSidecar":
            # If every slide is a video, skip; if any is image, keep
            if not any(
                not node.is_video for node in post.get_sidecar_nodes()
            ):
                continue
        elif post.typename == "GraphVideo" or post.is_video:
            continue
        yield post
        n += 1


def _build_loader() -> instaloader.Instaloader:
    L = instaloader.Instaloader(
        download_pictures=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
    )
    # Optional: log in once with `instaloader -l YOUR_USER`, then re-use session
    # or set env INSTALOADER_USER to a username that already has a stored session.
    u = os.environ.get("INSTALOADER_USER", "").strip()
    if u:
        try:
            L.load_session_from_file(u)
        except (FileNotFoundError, OSError):
            L.context.log(
                f"INSTALOADER_USER={u!r} but no session file found; use "
                f"`instaloader -l {u}` once or omit INSTALOADER_USER."
            )
    return L


@dataclass
class PostEntry:
    index: int
    shortcode: str
    taken_at: str
    url: str
    caption: str
    is_video: bool
    typename: str
    rel_paths: List[str]  # relative to influencer root
    likes: int
    comment_count: int


def download_influencer_last_photos(
    handle: str,
    *,
    base_dir: Path | str = "data",
    max_posts: int = 10,
    loader: Optional[instaloader.Instaloader] = None,
) -> dict[str, Any]:
    """
    Download the latest `max_posts` **photo** posts (no video-only) for
    @handle. Images go under: ``{base_dir}/influencers/{username}/posts/{idx}_{shortcode}/``.

    Returns a dict suitable for appending to your DB: username, base_path, posts[].
    """
    username = _normalize_username(handle)
    base = Path(base_dir)
    root = (base / "influencers" / username).resolve()
    posts_root = root / "posts"
    posts_root.mkdir(parents=True, exist_ok=True)

    L = loader or _build_loader()
    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except instaloader.exceptions.ProfileNotExistsException as e:
        raise RuntimeError(
            f"Could not load @{username}. The profile may be private, the "
            f"handle may be wrong, or Instagram blocked the request (HTTP 403). "
            f"Log in once with: instaloader -l YOUR_IG_USER "
            f"then re-run with env INSTALOADER_USER=YOUR_IG_USER"
        ) from e
    full_name = profile.full_name
    post_entries: list[PostEntry] = []
    for idx, post in enumerate(
        _iter_photo_posts(profile, max_posts), start=1
    ):
        post_dir = posts_root / f"{idx:02d}_{post.shortcode}"
        if post_dir.exists():
            for child in post_dir.iterdir():
                if child.is_file():
                    child.unlink()
        post_dir.mkdir(parents=True, exist_ok=True)

        # download_post writes images + *metadata* json into `target` directory
        L.download_post(post, target=str(post_dir))

        rel_paths: list[str] = []
        for f in sorted(post_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                rel_paths.append(
                    str(f.relative_to(root)).replace("\\", "/")
                )
        ent = PostEntry(
            index=idx,
            shortcode=post.shortcode,
            taken_at=post.date_local.isoformat(),
            url=f"https://www.instagram.com/p/{post.shortcode}/",
            caption=post.caption or "",
            is_video=bool(post.is_video),
            typename=str(post.typename or ""),
            rel_paths=rel_paths,
            likes=post.likes,
            comment_count=post.comments,
        )
        post_entries.append(ent)

    manifest: dict[str, Any] = {
        "username": username,
        "full_name": full_name,
        "followers": profile.followers,
        "ig_id": profile.userid,
        "base_path": str(root).replace("\\", "/"),
        "posts": [asdict(p) for p in post_entries],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base / "influencers_index.jsonl").parent.mkdir(
        parents=True, exist_ok=True
    )
    with (Path(base) / "influencers_index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ingest.download_instagram <@username|username> "
            "[--base-dir data] [--max-posts 10]",
            file=sys.stderr,
        )
        return 1
    argv = [a for a in sys.argv[1:] if a]
    handle = argv[0]
    base_dir = "data"
    max_posts = 10
    i = 1
    while i < len(argv):
        if argv[i] == "--base-dir" and i + 1 < len(argv):
            base_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--max-posts" and i + 1 < len(argv):
            max_posts = int(argv[i + 1])
            i += 2
        else:
            i += 1
    manifest = download_influencer_last_photos(
        handle, base_dir=base_dir, max_posts=max_posts
    )
    print(
        json.dumps(
            {
                "username": manifest["username"],
                "saved_posts": len(manifest["posts"]),
                "root": manifest["base_path"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

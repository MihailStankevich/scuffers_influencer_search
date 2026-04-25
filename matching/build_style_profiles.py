"""
Compute CLIP embeddings for influencer images and build style profiles per influencer.

Inputs:
  - data/influencers/<username>/images/*
  - data/influencers/influencers_geo.json (to restrict to selected influencers)

Outputs:
  - data/influencers/<username>/style_profile.json
  - data/influencers/style_profiles_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from ingest.paths import DATA_INFLUENCERS_DIR, INFLUENCERS_GEO_JSON

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _load_geo_usernames() -> list[str]:
    geo = json.loads(INFLUENCERS_GEO_JSON.read_text(encoding="utf-8"))
    if not isinstance(geo, dict):
        raise SystemExit("influencers_geo.json must be a JSON object")
    return sorted(k.strip().lower() for k in geo.keys() if k.strip())


def _image_paths_for_user(username: str) -> list[Path]:
    folder = DATA_INFLUENCERS_DIR / username / "images"
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def _encode_image(
    image_path: Path, model: torch.nn.Module, preprocess: Any, device: str
) -> np.ndarray | None:
    try:
        img = Image.open(image_path).convert("RGB")
    except (OSError, ValueError):
        return None
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    arr = emb[0].detach().cpu().numpy().astype(np.float32)
    return arr


def run(model_name: str, pretrained: str, device: str) -> None:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()

    usernames = _load_geo_usernames()
    index_rows: list[dict[str, Any]] = []
    skipped_users: list[str] = []

    for username in tqdm(usernames, desc="Building style profiles"):
        images = _image_paths_for_user(username)
        if not images:
            skipped_users.append(username)
            continue

        vectors: list[np.ndarray] = []
        used_images: list[str] = []
        image_embeddings: list[dict[str, Any]] = []
        for img in images:
            vec = _encode_image(img, model, preprocess, device)
            if vec is None:
                continue
            rel = str(img.relative_to(DATA_INFLUENCERS_DIR)).replace("\\", "/")
            vectors.append(vec)
            used_images.append(rel)
            image_embeddings.append(
                {
                    "path": rel,
                    "embedding": vec.tolist(),  # already L2-normalized
                }
            )

        if not vectors:
            skipped_users.append(username)
            continue

        mat = np.vstack(vectors)
        centroid = _l2_normalize(mat.mean(axis=0))
        profile = {
            "username": username,
            "model_name": model_name,
            "pretrained": pretrained,
            "embedding_dim": int(centroid.shape[0]),
            "image_count": len(vectors),
            "images": used_images,
            "centroid": centroid.tolist(),  # normalized centroid
            "image_embeddings": image_embeddings,
        }
        out = DATA_INFLUENCERS_DIR / username / "style_profile.json"
        out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "username": username,
                "image_count": len(vectors),
                "style_profile": str(out.relative_to(DATA_INFLUENCERS_DIR)).replace("\\", "/"),
            }
        )

    idx = {
        "model_name": model_name,
        "pretrained": pretrained,
        "count_profiles": len(index_rows),
        "count_skipped": len(skipped_users),
        "skipped_usernames": skipped_users,
        "profiles": index_rows,
    }
    (DATA_INFLUENCERS_DIR / "style_profiles_index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built profiles: {len(index_rows)}")
    print(f"Skipped (no usable images): {len(skipped_users)}")
    print(f"Index: {DATA_INFLUENCERS_DIR / 'style_profiles_index.json'}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build CLIP style profiles per influencer")
    p.add_argument("--model", default="ViT-H-14", help="open_clip model name")
    p.add_argument(
        "--pretrained",
        default="laion2b_s32b_b79k",
        help="open_clip pretrained checkpoint name",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = p.parse_args()
    run(model_name=args.model, pretrained=args.pretrained, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

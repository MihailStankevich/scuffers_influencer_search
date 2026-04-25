"""
Match a product image to influencer style centroids.

Primary signal: CLIP visual similarity.
Optional reranking: geo, engagement, keyword/topic.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

from ingest.paths import DATA_INFLUENCERS_DIR, INFLUENCERS_GEO_JSON

STYLE_INDEX = DATA_INFLUENCERS_DIR / "style_profiles_index.json"
FEATURES_DB = DATA_INFLUENCERS_DIR / "influencer_features.db"
FEATURES_JSON = DATA_INFLUENCERS_DIR / "influencer_features.json"

_clip_lock = threading.Lock()
_clip_cache: dict[tuple[str, str, str], tuple[torch.nn.Module, Any]] = {}


def get_clip_model(
    model_name: str, pretrained: str, device: str
) -> tuple[torch.nn.Module, Any]:
    """Load OpenCLIP once per (model, checkpoint, device); reuse for all matches."""
    key = (model_name, pretrained, device)
    with _clip_lock:
        hit = _clip_cache.get(key)
        if hit is not None:
            return hit
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, device=device
        )
        model.eval()
        _clip_cache[key] = (model, preprocess)
        return model, preprocess


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _encode_product(
    image_path: Path, model: torch.nn.Module, preprocess: Any, device: str
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].detach().cpu().numpy().astype(np.float32)


def _load_geo() -> dict[str, dict[str, Any]]:
    geo = json.loads(INFLUENCERS_GEO_JSON.read_text(encoding="utf-8"))
    if not isinstance(geo, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in geo.items():
        if isinstance(v, dict):
            out[k.lower()] = v
    return out


def _load_features() -> dict[str, dict[str, Any]]:
    if FEATURES_DB.is_file():
        with sqlite3.connect(FEATURES_DB) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT username, avg_likes, avg_comments, top_hashtags_json
                FROM influencer_features
                """
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for username, avg_likes, avg_comments, hashtags_json in rows:
            try:
                tags = json.loads(hashtags_json or "[]")
            except json.JSONDecodeError:
                tags = []
            out[str(username).lower()] = {
                "avg_likes": float(avg_likes or 0.0),
                "avg_comments": float(avg_comments or 0.0),
                "top_hashtags": tags if isinstance(tags, list) else [],
            }
        return out

    if FEATURES_JSON.is_file():
        payload = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        out = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            u = str(r.get("username", "")).lower().strip()
            if not u:
                continue
            out[u] = {
                "avg_likes": float(r.get("avg_likes") or 0.0),
                "avg_comments": float(r.get("avg_comments") or 0.0),
                "top_hashtags": r.get("top_hashtags") or [],
            }
        return out
    return {}


def _load_profiles() -> list[dict[str, Any]]:
    idx = json.loads(STYLE_INDEX.read_text(encoding="utf-8"))
    profiles = idx.get("profiles", [])
    if not isinstance(profiles, list):
        return []
    out = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        rel = p.get("style_profile")
        u = str(p.get("username", "")).lower().strip()
        if not rel or not u:
            continue
        profile_path = DATA_INFLUENCERS_DIR / rel
        if not profile_path.is_file():
            continue
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        centroid = payload.get("centroid")
        if not isinstance(centroid, list) or not centroid:
            continue
        img_embs: list[np.ndarray] = []
        raw_img_embs = payload.get("image_embeddings")
        if isinstance(raw_img_embs, list):
            for it in raw_img_embs:
                if not isinstance(it, dict):
                    continue
                emb = it.get("embedding")
                if isinstance(emb, list) and emb:
                    img_embs.append(np.array(emb, dtype=np.float32))
        out.append(
            {
                "username": u,
                "centroid": np.array(centroid, dtype=np.float32),
                "image_embeddings": img_embs,
            }
        )
    return out


def _geo_score(geo_data: dict[str, Any], country: str | None, city: str | None) -> float:
    if not country and not city:
        return 0.0
    g_country = str(geo_data.get("country") or "").strip().lower()
    g_city = str(geo_data.get("city") or "").strip().lower()
    c = (country or "").strip().lower()
    ci = (city or "").strip().lower()
    if ci and g_city and ci == g_city:
        return 1.0
    if c and g_country and c == g_country:
        return 0.6
    return 0.0


def _safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def _engagement_raw(feat: dict[str, Any]) -> float:
    likes = float(feat.get("avg_likes") or 0.0)
    comments = float(feat.get("avg_comments") or 0.0)
    return _safe_log1p(likes) + 1.5 * _safe_log1p(comments)


def _topic_score(feat: dict[str, Any], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    tags = feat.get("top_hashtags") or []
    top = {str(t.get("tag", "")).lower() for t in tags if isinstance(t, dict)}
    if not top:
        return 0.0
    kw = {k.strip().lower().lstrip("#") for k in keywords if k.strip()}
    if not kw:
        return 0.0
    overlap = len(top.intersection(kw))
    return min(1.0, overlap / max(1, min(5, len(kw))))


def _style_similarity(
    product_vec: np.ndarray,
    profile: dict[str, Any],
    pooling_mode: str,
    image_top_k: int,
) -> float:
    """
    Returns style similarity in [0,1] (normalized from cosine).
    pooling_mode:
      - centroid: cosine(product, centroid)
      - max: max cosine(product, each_image_embedding)
      - topk_mean: mean of top-k image-level cosine values
    """
    mode = (pooling_mode or "topk_mean").strip().lower()
    centroid = _l2_normalize(np.array(profile["centroid"], dtype=np.float32))
    d = int(product_vec.shape[0])
    if int(centroid.shape[0]) != d:
        return 0.0

    if mode == "centroid":
        cos = float(np.dot(product_vec, centroid))
        return (cos + 1.0) / 2.0

    img_embs: list[np.ndarray] = profile.get("image_embeddings") or []
    if not img_embs:
        # Backward compatibility with old profiles that only store centroid
        cos = float(np.dot(product_vec, centroid))
        return (cos + 1.0) / 2.0

    compatible = [v for v in img_embs if int(v.shape[0]) == d]
    sims = [float(np.dot(product_vec, _l2_normalize(v))) for v in compatible]
    if not sims:
        cos = float(np.dot(product_vec, centroid))
        return (cos + 1.0) / 2.0

    if mode == "max":
        cos = max(sims)
    else:
        k = max(1, min(int(image_top_k), len(sims)))
        cos = float(np.mean(sorted(sims, reverse=True)[:k]))
    return (cos + 1.0) / 2.0


def run_match(
    product_image: Path,
    top_k: int,
    country: str | None,
    city: str | None,
    keywords: list[str],
    w_style: float,
    w_geo: float,
    w_engagement: float,
    w_topic: float,
    device: str,
    pooling_mode: str = "topk_mean",
    image_top_k: int = 2,
    country_mandatory: bool = False,
    city_mandatory: bool = False,
    min_style_for_business: float = 0.60,
) -> dict[str, Any]:
    if not STYLE_INDEX.is_file():
        raise SystemExit("Missing style profiles. Run: py -m matching.build_style_profiles")
    profiles = _load_profiles()
    if not profiles:
        raise SystemExit("No style profiles available.")

    idx = json.loads(STYLE_INDEX.read_text(encoding="utf-8"))
    model_name = idx.get("model_name", "ViT-H-14")
    pretrained = idx.get("pretrained", "laion2b_s32b_b79k")
    model, preprocess = get_clip_model(model_name, pretrained, device)

    prod = _encode_product(product_image, model, preprocess, device)
    prod = _l2_normalize(prod)

    geo_map = _load_geo()
    feat_map = _load_features()

    # Engagement as percentile rank is more robust than min-max against outliers.
    raw_by_user: dict[str, float] = {}
    for p in profiles:
        u = p["username"]
        raw_by_user[u] = _engagement_raw(feat_map.get(u, {}))
    ordered_users = [u for u, _ in sorted(raw_by_user.items(), key=lambda x: x[1])]
    if len(ordered_users) > 1:
        e_rank = {u: i / (len(ordered_users) - 1) for i, u in enumerate(ordered_users)}
    else:
        e_rank = {u: 0.0 for u in ordered_users}

    scored = []

    for p in profiles:
        username = p["username"]
        cent_dim = int(np.array(p["centroid"]).shape[0])
        if cent_dim != int(prod.shape[0]):
            continue
        style_norm = _style_similarity(
            product_vec=prod,
            profile=p,
            pooling_mode=pooling_mode,
            image_top_k=image_top_k,
        )
        style = (2.0 * style_norm) - 1.0

        geo_data = geo_map.get(username, {})
        g = _geo_score(geo_data, country=country, city=city)

        # Optional strict geo filters for pitch scenarios.
        if country_mandatory and country and g <= 0.0:
            continue
        if city_mandatory and city:
            g_city = str(geo_data.get("city") or "").strip().lower()
            if not g_city or g_city != str(city).strip().lower():
                continue

        feat = feat_map.get(username, {})
        e = float(e_rank.get(username, 0.0))
        t = _topic_score(feat, keywords)

        # Keep style as dominant signal: if style is too low, disable secondary boosts.
        if style_norm < float(min_style_for_business):
            e = 0.0
            t = 0.0

        final = (
            w_style * style_norm + w_geo * g + w_engagement * e + w_topic * t
        )
        scored.append(
            {
                "username": username,
                "cosine_similarity": round(style, 6),
                "style_score": round(style_norm, 6),
                "style_match_percent": round(style_norm * 100.0, 2),
                "geo_score": round(g, 6),
                "engagement_score": round(e, 6),
                "topic_score": round(t, 6),
                "final_score": round(float(final), 6),
                "final_match_percent": round(float(final) * 100.0, 2),
                "country": geo_data.get("country"),
                "city": geo_data.get("city"),
                "avg_likes": round(float(feat.get("avg_likes") or 0.0), 3),
                "avg_comments": round(float(feat.get("avg_comments") or 0.0), 3),
                "top_hashtags": feat.get("top_hashtags") or [],
            }
        )

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return {
        "product_image": str(product_image),
        "model_name": model_name,
        "pretrained": pretrained,
        "weights": {
            "style": w_style,
            "geo": w_geo,
            "engagement": w_engagement,
            "topic": w_topic,
        },
        "pooling": {"mode": pooling_mode, "image_top_k": image_top_k},
        "filters": {
            "country": country,
            "city": city,
            "keywords": keywords,
            "country_mandatory": country_mandatory,
            "city_mandatory": city_mandatory,
            "min_style_for_business": min_style_for_business,
        },
        "top_k": scored[:top_k],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Match a product image against influencer styles")
    p.add_argument("--image", type=Path, required=True, help="Product image path")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--country", default=None)
    p.add_argument("--city", default=None)
    p.add_argument(
        "--keywords",
        default="",
        help="Comma-separated keywords/hashtags, e.g. oversized,minimal,sneakers",
    )
    p.add_argument("--w-style", type=float, default=0.70)
    p.add_argument("--w-geo", type=float, default=0.15)
    p.add_argument("--w-engagement", type=float, default=0.10)
    p.add_argument("--w-topic", type=float, default=0.05)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    p.add_argument(
        "--pooling-mode",
        default="topk_mean",
        choices=["centroid", "max", "topk_mean"],
    )
    p.add_argument("--image-top-k", type=int, default=2)
    args = p.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]
    payload = run_match(
        product_image=args.image,
        top_k=max(1, args.top_k),
        country=args.country,
        city=args.city,
        keywords=keywords,
        w_style=args.w_style,
        w_geo=args.w_geo,
        w_engagement=args.w_engagement,
        w_topic=args.w_topic,
        device=args.device,
        pooling_mode=args.pooling_mode,
        image_top_k=max(1, args.image_top_k),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

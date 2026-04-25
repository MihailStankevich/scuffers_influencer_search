"""
Visualize product vs influencer style space for demo/pitch.

Outputs:
  - data/influencers/visualizations/product_match_pca.png
  - data/influencers/visualizations/product_match_pca_3d.png
  - data/influencers/visualizations/product_match_top10_local_pca.png
  - data/influencers/visualizations/product_match_top10_similarity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ingest.paths import DATA_INFLUENCERS_DIR
from matching.match_product import (
    _encode_product,
    _load_profiles,
    get_clip_model,
    run_match,
)

import torch


STYLE_INDEX = DATA_INFLUENCERS_DIR / "style_profiles_index.json"
OUT_PLOT = DATA_INFLUENCERS_DIR / "visualizations" / "product_match_pca.png"
OUT_PLOT_3D = DATA_INFLUENCERS_DIR / "visualizations" / "product_match_pca_3d.png"
OUT_LOCAL_PLOT = DATA_INFLUENCERS_DIR / "visualizations" / "product_match_top10_local_pca.png"
OUT_SIM_PLOT = DATA_INFLUENCERS_DIR / "visualizations" / "product_match_top10_similarity.png"


def _get_model_config() -> tuple[str, str]:
    idx = json.loads(STYLE_INDEX.read_text(encoding="utf-8"))
    model_name = idx.get("model_name", "ViT-H-14")
    pretrained = idx.get("pretrained", "laion2b_s32b_b79k")
    return str(model_name), str(pretrained)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _plot_global_pca(
    Z_inf: np.ndarray,
    z_prod: np.ndarray,
    labels: list[str],
    top_set: set[str],
    title_suffix: str,
    explained_var_2d: float,
) -> None:
    plt.figure(figsize=(14, 10))
    plt.scatter(Z_inf[:, 0], Z_inf[:, 1], s=45, alpha=0.5, c="#9aa0a6", label="Influencers")

    top_points = [(i, u) for i, u in enumerate(labels) if u in top_set]
    if top_points:
        idxs = [i for i, _ in top_points]
        plt.scatter(
            Z_inf[idxs, 0],
            Z_inf[idxs, 1],
            s=95,
            alpha=0.95,
            c="#1f77b4",
            label=f"Top {len(top_points)} influencers",
        )
        for i, u in top_points:
            plt.annotate(u, (Z_inf[i, 0], Z_inf[i, 1]), fontsize=9, color="#174a7a")

    plt.scatter(
        [z_prod[0]],
        [z_prod[1]],
        s=200,
        c="#d62728",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="Uploaded product",
        zorder=10,
    )
    plt.annotate("PRODUCT", (z_prod[0], z_prod[1]), fontsize=10, color="#8c1d18")

    title = "Product vs Influencer Style Space (Global PCA)"
    details = [f"2D explained variance={explained_var_2d:.1%}"]
    if title_suffix:
        details.append(title_suffix)
    plt.title(title + "\n" + " | ".join(details))
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=190)
    plt.close()


def _plot_local_topk_pca(
    product_vec: np.ndarray,
    profiles_by_user: dict[str, np.ndarray],
    top_usernames: list[str],
) -> None:
    vecs = [product_vec]
    labels = ["PRODUCT"]
    for u in top_usernames:
        vec = profiles_by_user.get(u)
        if vec is not None:
            vecs.append(vec)
            labels.append(u)
    X_local = np.vstack(vecs)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_local)
    evr = float(pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(11, 8))
    plt.scatter(Z[1:, 0], Z[1:, 1], s=95, c="#1f77b4", alpha=0.9, label="Top influencers")
    for i in range(1, len(labels)):
        plt.annotate(labels[i], (Z[i, 0], Z[i, 1]), fontsize=9, color="#174a7a")
    plt.scatter(
        [Z[0, 0]],
        [Z[0, 1]],
        s=240,
        c="#d62728",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="Uploaded product",
        zorder=10,
    )
    plt.annotate("PRODUCT", (Z[0, 0], Z[0, 1]), fontsize=10, color="#8c1d18")
    plt.title(
        "Local neighborhood PCA (PRODUCT + Top influencers)\n"
        f"2D explained variance={evr:.1%}"
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_LOCAL_PLOT, dpi=190)
    plt.close()


def _plot_global_pca_3d(
    X_all: np.ndarray,
    labels: list[str],
    top_set: set[str],
) -> None:
    # 3D PCA offers a less lossy visual than 2D, still approximate.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pca3 = PCA(n_components=3, random_state=42)
    Z3 = pca3.fit_transform(X_all)
    evr3 = float(pca3.explained_variance_ratio_.sum())
    Z_inf = Z3[:-1]
    z_prod = Z3[-1]

    fig = plt.figure(figsize=(8.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        Z_inf[:, 0],
        Z_inf[:, 1],
        Z_inf[:, 2],
        s=32,
        alpha=0.35,
        c="#9aa0a6",
        label="Influencers",
    )
    top_points = [(i, u) for i, u in enumerate(labels) if u in top_set]
    if top_points:
        idxs = [i for i, _ in top_points]
        ax.scatter(
            Z_inf[idxs, 0],
            Z_inf[idxs, 1],
            Z_inf[idxs, 2],
            s=78,
            alpha=0.95,
            c="#1f77b4",
            label=f"Top {len(top_points)} influencers",
        )
    ax.scatter(
        [z_prod[0]],
        [z_prod[1]],
        [z_prod[2]],
        s=220,
        c="#d62728",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="PRODUCT",
    )
    ax.set_title(f"Product vs influencers in 3D PCA (explained variance={evr3:.1%})")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PLOT_3D, dpi=170)
    plt.close(fig)


def _plot_topk_similarity_bars(product_vec: np.ndarray, profiles_by_user: dict[str, np.ndarray], top_usernames: list[str]) -> None:
    rows = []
    for u in top_usernames:
        vec = profiles_by_user.get(u)
        if vec is None:
            continue
        sim = float(np.dot(product_vec, vec))
        rows.append((u, sim))
    rows.sort(key=lambda x: x[1], reverse=True)
    if not rows:
        return
    names = [r[0] for r in rows]
    sims = [r[1] for r in rows]

    plt.figure(figsize=(8.4, 3.6))
    y = np.arange(len(names))
    plt.barh(y, sims, color="#1f77b4", alpha=0.9)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    for i, v in enumerate(sims):
        style_norm = (v + 1.0) / 2.0
        plt.text(v + 0.005, i, f"cos={v:.3f} | style={style_norm:.3f}", va="center", fontsize=8)
    plt.title("Top influencer neighbors by cosine similarity (original CLIP space)")
    plt.xlabel("Raw cosine similarity to PRODUCT centroid (style_score = (cos + 1) / 2)")
    plt.xlim(min(0.0, min(sims) - 0.05), min(1.0, max(sims) + 0.08))
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_SIM_PLOT, dpi=170)
    plt.close()


def run(
    image_path: Path,
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
) -> None:
    profiles = _load_profiles()
    if not profiles:
        raise SystemExit("No style profiles available. Run matching.build_style_profiles first.")

    # Get top-k rankings (same scoring pipeline as backend)
    ranking = run_match(
        product_image=image_path,
        top_k=top_k,
        country=country,
        city=city,
        keywords=keywords,
        w_style=w_style,
        w_geo=w_geo,
        w_engagement=w_engagement,
        w_topic=w_topic,
        device=device,
        pooling_mode=pooling_mode,
        image_top_k=image_top_k,
        country_mandatory=country_mandatory,
        city_mandatory=city_mandatory,
        min_style_for_business=min_style_for_business,
    )
    top_usernames = [r["username"] for r in ranking["top_k"]]
    top_set = set(top_usernames)

    # Encode product with same CLIP config (reuse cached model from run_match when possible)
    model_name, pretrained = _get_model_config()
    model, preprocess = get_clip_model(model_name, pretrained, device)
    prod_vec = _l2_normalize(_encode_product(image_path, model, preprocess, device))

    labels = [p["username"] for p in profiles]
    X_centroids = np.vstack([_l2_normalize(p["centroid"]) for p in profiles])
    profiles_by_user = {p["username"]: _l2_normalize(p["centroid"]) for p in profiles}
    X = np.vstack([X_centroids, prod_vec])

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    explained_var_2d = float(pca.explained_variance_ratio_.sum())
    Z_inf = Z[:-1]
    z_prod = Z[-1]

    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    title_suffix = []
    if city:
        title_suffix.append(f"city={city}")
    if country:
        title_suffix.append(f"country={country}")
    if keywords:
        title_suffix.append("keywords=" + ",".join(keywords))
    suffix = " | ".join(title_suffix)
    _plot_global_pca(
        Z_inf=Z_inf,
        z_prod=z_prod,
        labels=labels,
        top_set=top_set,
        title_suffix=suffix,
        explained_var_2d=explained_var_2d,
    )
    _plot_local_topk_pca(
        product_vec=prod_vec,
        profiles_by_user=profiles_by_user,
        top_usernames=top_usernames,
    )
    _plot_global_pca_3d(
        X_all=X,
        labels=labels,
        top_set=top_set,
    )
    _plot_topk_similarity_bars(
        product_vec=prod_vec,
        profiles_by_user=profiles_by_user,
        top_usernames=top_usernames,
    )

    # Save ranking side-by-side for traceability in demo
    out_json = OUT_PLOT.with_suffix(".json")
    out_json.write_text(json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PLOT}")
    print(f"Saved: {OUT_PLOT_3D}")
    print(f"Saved: {OUT_LOCAL_PLOT}")
    print(f"Saved: {OUT_SIM_PLOT}")
    print(f"Saved: {out_json}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot uploaded product among influencer centroids")
    p.add_argument("--image", type=Path, required=True, help="Product image path")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--country", default=None)
    p.add_argument("--city", default=None)
    p.add_argument("--keywords", default="", help="Comma-separated keywords")
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
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    run(
        image_path=args.image,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


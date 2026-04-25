"""
Visualize influencer style centroids in 2D and as similarity heatmap.

Reads:
  - data/influencers/style_profiles_index.json
  - data/influencers/<username>/style_profile.json

Writes:
  - data/influencers/visualizations/style_centroids_pca.png
  - data/influencers/visualizations/style_centroids_tsne.png
  - data/influencers/visualizations/style_similarity_heatmap.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ingest.paths import DATA_INFLUENCERS_DIR

STYLE_INDEX = DATA_INFLUENCERS_DIR / "style_profiles_index.json"
OUT_DIR = DATA_INFLUENCERS_DIR / "visualizations"


def _load_centroids() -> tuple[list[str], np.ndarray]:
    if not STYLE_INDEX.is_file():
        raise SystemExit(
            "Missing style_profiles_index.json. Run `uv run python -m matching.build_style_profiles` first."
        )
    idx = json.loads(STYLE_INDEX.read_text(encoding="utf-8"))
    profiles = idx.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("No profiles in style_profiles_index.json")

    names: list[str] = []
    vectors: list[np.ndarray] = []
    for item in profiles:
        if not isinstance(item, dict):
            continue
        username = str(item.get("username", "")).strip().lower()
        rel = item.get("style_profile")
        if not username or not rel:
            continue
        p = DATA_INFLUENCERS_DIR / str(rel)
        if not p.is_file():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        centroid = payload.get("centroid")
        if not isinstance(centroid, list) or not centroid:
            continue
        names.append(username)
        vectors.append(np.array(centroid, dtype=np.float32))

    if not vectors:
        raise SystemExit("No valid centroid vectors found.")
    mat = np.vstack(vectors)
    return names, mat


def _plot_scatter(coords: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(13, 9))
    plt.scatter(coords[:, 0], coords[:, 1], s=55, alpha=0.85)
    for i, name in enumerate(labels):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.9)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_heatmap(sim: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig_w = max(10, len(labels) * 0.45)
    fig_h = max(8, len(labels) * 0.4)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Cosine similarity")
    plt.title("Influencer style centroid similarity")
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(labels)), labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(perplexity: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, X = _load_centroids()

    # PCA (deterministic, fast)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    _plot_scatter(
        X_pca,
        labels,
        "Influencer style space (PCA of CLIP centroids)",
        OUT_DIR / "style_centroids_pca.png",
    )

    # t-SNE (preserves local neighborhoods better)
    # Perplexity must be < n_samples
    n = X.shape[0]
    p = min(perplexity, max(5.0, float(n - 1)))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=p,
        random_state=42,
    )
    X_tsne = tsne.fit_transform(X)
    _plot_scatter(
        X_tsne,
        labels,
        "Influencer style space (t-SNE of CLIP centroids)",
        OUT_DIR / "style_centroids_tsne.png",
    )

    # Similarity matrix among centroids
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = Xn @ Xn.T
    _plot_heatmap(sim, labels, OUT_DIR / "style_similarity_heatmap.png")

    print(f"Saved: {OUT_DIR / 'style_centroids_pca.png'}")
    print(f"Saved: {OUT_DIR / 'style_centroids_tsne.png'}")
    print(f"Saved: {OUT_DIR / 'style_similarity_heatmap.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize influencer style centroids")
    parser.add_argument(
        "--perplexity",
        type=float,
        default=10.0,
        help="t-SNE perplexity (auto-clipped by sample size)",
    )
    args = parser.parse_args()
    run(perplexity=args.perplexity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


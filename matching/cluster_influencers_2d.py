"""
Create a 2D clustered view of influencer style centroids.

Outputs:
  - data/influencers/visualizations/style_clusters_2d.png
  - data/influencers/visualizations/style_clusters_2d.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ingest.paths import DATA_INFLUENCERS_DIR

STYLE_INDEX = DATA_INFLUENCERS_DIR / "style_profiles_index.json"
OUT_DIR = DATA_INFLUENCERS_DIR / "visualizations"
OUT_PNG = OUT_DIR / "style_clusters_2d.png"
OUT_JSON = OUT_DIR / "style_clusters_2d.json"


def _load_centroids() -> tuple[list[str], np.ndarray]:
    idx = json.loads(STYLE_INDEX.read_text(encoding="utf-8"))
    profiles = idx.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("No profiles found in style_profiles_index.json")

    names: list[str] = []
    vecs: list[np.ndarray] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        u = str(p.get("username", "")).strip().lower()
        rel = p.get("style_profile")
        if not u or not rel:
            continue
        f = DATA_INFLUENCERS_DIR / str(rel)
        if not f.is_file():
            continue
        payload = json.loads(f.read_text(encoding="utf-8"))
        c = payload.get("centroid")
        if not isinstance(c, list) or not c:
            continue
        names.append(u)
        vecs.append(np.array(c, dtype=np.float32))

    if not vecs:
        raise SystemExit("No valid centroid vectors found.")
    return names, np.vstack(vecs)


def run(k: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names, X = _load_centroids()

    if k < 2:
        raise SystemExit("k must be >= 2")
    if k > len(names):
        raise SystemExit(f"k={k} > number of influencers={len(names)}")

    # Cluster in original space, visualize in 2D PCA
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_ids = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    evr = float(pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab10", k)
    for cid in range(k):
        mask = cluster_ids == cid
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=70,
            alpha=0.9,
            color=cmap(cid),
            label=f"Cluster {cid}",
        )
    for i, u in enumerate(names):
        plt.annotate(u, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.9)
    plt.title(f"Influencer style clusters (KMeans k={k}) on PCA 2D (variance={evr:.1%})")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close()

    rows = []
    for i, u in enumerate(names):
        rows.append(
            {
                "username": u,
                "cluster": int(cluster_ids[i]),
                "x": float(Z[i, 0]),
                "y": float(Z[i, 1]),
            }
        )

    payload = {
        "k": k,
        "count_influencers": len(names),
        "pca_2d_explained_variance": evr,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_JSON}")


def main() -> int:
    p = argparse.ArgumentParser(description="2D cluster view of influencer style centroids")
    p.add_argument("--k", type=int, default=4, help="Number of KMeans clusters")
    args = p.parse_args()
    run(k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


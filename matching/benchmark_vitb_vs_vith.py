"""
Benchmark comparison: ViT-B/32 vs ViT-H/14 on 6 fixed test images.

Produces:
  - data/influencers/visualizations/model_compare_vitb_vs_vith.json
  - data/influencers/visualizations/model_compare_vitb_vs_vith.md

Notes:
  - Rebuilds style profiles for each model to ensure fair comparison.
  - Uses style-only ranking with identical pooling config.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from ingest.paths import DATA_INFLUENCERS_DIR, REPO_ROOT
from matching.build_style_profiles import run as build_profiles
from matching.match_product import run_match

OUT_DIR = DATA_INFLUENCERS_DIR / "visualizations"
OUT_JSON = OUT_DIR / "model_compare_vitb_vs_vith.json"
OUT_MD = OUT_DIR / "model_compare_vitb_vs_vith.md"


def _test_images() -> list[tuple[str, Path]]:
    # Cursor attaches user-provided assets here in this environment.
    base = (
        Path.home()
        / ".cursor"
        / "projects"
        / "c-Users-mihai-Documents-GitHub-scuffers"
        / "assets"
    )
    items = [
        ("puffer", base / "c__Users_mihai_Documents_GitHub_scuffers_puffer.png"),
        ("eme", base / "c__Users_mihai_Documents_GitHub_scuffers_eme_sudadera.jpg"),
        ("street2", base / "c__Users_mihai_Documents_GitHub_scuffers_street2.jpg"),
        ("streetwear", base / "c__Users_mihai_Documents_GitHub_scuffers_streetwear.jpg"),
        ("suit", base / "c__Users_mihai_Documents_GitHub_scuffers_Suit.png"),
        ("zps0047", base / "c__Users_mihai_Documents_GitHub_scuffers_ZPS0047.jpg"),
    ]
    return [(name, p) for name, p in items if p.is_file()]


def _run_eval_for_model(model_name: str, pretrained: str, top_k: int) -> dict[str, Any]:
    # Rebuild influencer style profiles with the requested model.
    build_profiles(model_name=model_name, pretrained=pretrained, device="cpu")

    rows: list[dict[str, Any]] = []
    for label, path in _test_images():
        result = run_match(
            product_image=path,
            top_k=top_k,
            country=None,
            city=None,
            keywords=[],
            w_style=1.0,
            w_geo=0.0,
            w_engagement=0.0,
            w_topic=0.0,
            device="cpu",
            pooling_mode="topk_mean",
            image_top_k=2,
            country_mandatory=False,
            city_mandatory=False,
            min_style_for_business=0.0,
        )
        top = result.get("top_k", [])
        top1 = top[0] if top else {}
        rows.append(
            {
                "image_label": label,
                "image_path": str(path),
                "top1_username": top1.get("username"),
                "top1_style_score": float(top1.get("style_score") or 0.0),
                "top1_style_percent": float(top1.get("style_match_percent") or 0.0),
                "topk_usernames": [x.get("username") for x in top if isinstance(x, dict)],
                "topk_style_percent": [
                    float(x.get("style_match_percent") or 0.0) for x in top if isinstance(x, dict)
                ],
            }
        )
    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "pooling_mode": "topk_mean",
        "image_top_k": 2,
        "top_k": top_k,
        "rows": rows,
    }


def _build_markdown(vitb: dict[str, Any], vith: dict[str, Any]) -> str:
    rows_b = {r["image_label"]: r for r in vitb["rows"]}
    rows_h = {r["image_label"]: r for r in vith["rows"]}

    lines = []
    lines.append("# ViT-B/32 vs ViT-H/14 (Style-only)\\n")
    lines.append(
        f"- ViT-B: `{vitb['model_name']}` + `{vitb['pretrained']}`  \n"
        f"- ViT-H: `{vith['model_name']}` + `{vith['pretrained']}`  \n"
        f"- Pooling: `{vitb['pooling_mode']}`, `image_top_k={vitb['image_top_k']}`  \n"
        f"- Evaluation top-k: `{vitb['top_k']}`\\n"
    )
    k = int(vitb.get("top_k") or 5)
    lines.append(
        f"| Image | ViT-B top1 | ViT-B style % | ViT-H top1 | ViT-H style % | Delta (H-B) | Top-{k} overlap |\n"
        "|---|---|---:|---|---:|---:|---:|"
    )

    for label in sorted(rows_b.keys()):
        rb = rows_b[label]
        rh = rows_h.get(label, {})
        b_user = rb.get("top1_username") or "-"
        h_user = rh.get("top1_username") or "-"
        b_pct = float(rb.get("top1_style_percent") or 0.0)
        h_pct = float(rh.get("top1_style_percent") or 0.0)
        delta = h_pct - b_pct
        b_top = set(rb.get("topk_usernames") or [])
        h_top = set(rh.get("topk_usernames") or [])
        overlap = len(b_top.intersection(h_top))
        lines.append(
            f"| {label} | {b_user} | {b_pct:.2f} | {h_user} | {h_pct:.2f} | {delta:+.2f} | {overlap}/{k} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ViT-B vs ViT-H on fixed test set")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = _test_images()
    if not tests:
        raise SystemExit("No test images found in assets folder.")

    top_k = max(1, int(args.top_k))
    vitb = _run_eval_for_model("ViT-B-32", "laion2b_s34b_b79k", top_k=top_k)
    vith = _run_eval_for_model("ViT-H-14", "laion2b_s32b_b79k", top_k=top_k)

    payload = {"vitb": vitb, "vith": vith}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_build_markdown(vitb, vith), encoding="utf-8")

    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


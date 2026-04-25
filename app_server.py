"""
FastAPI + Gradio app for Scuffers Creator Match AI.

Endpoints:
  - GET  /health
  - GET  /profiles
  - POST /match
  - POST /visualize
  - UI at /ui
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ingest.paths import DATA_INFLUENCERS_DIR, INFLUENCERS_GEO_JSON
from matching.match_product import _load_features, _load_profiles, run_match
from matching.visualize_product_match import (
    OUT_LOCAL_PLOT,
    OUT_PLOT,
    OUT_PLOT_3D,
    OUT_SIM_PLOT,
    run as run_visualize,
)

VIS_DIR = DATA_INFLUENCERS_DIR / "visualizations"
UPLOAD_DIR = VIS_DIR / "uploads"
RUNS_DIR = VIS_DIR / "runs"
TOP_K_FIXED = 3


def _mode_weights(mode: str) -> tuple[float, float, float, float]:
    m = (mode or "").strip().lower()
    if m == "style-only":
        return 1.0, 0.0, 0.0, 0.0
    if m == "business":
        return 0.70, 0.15, 0.10, 0.05
    raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _representative_image_for_username(username: str) -> str | None:
    folder = DATA_INFLUENCERS_DIR / username / "images"
    if not folder.is_dir():
        return None
    candidates = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not candidates:
        return None
    return str(candidates[0])


def _top_gallery_from_ranking(ranking: dict[str, Any]) -> list[tuple[str, str]]:
    rows = ranking.get("top_k", []) if isinstance(ranking, dict) else []
    gallery: list[tuple[str, str]] = []
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        username = str(row.get("username") or "").strip()
        if not username:
            continue
        img = _representative_image_for_username(username)
        if not img:
            continue
        style = row.get("style_score")
        final = row.get("final_score")
        caption = f"#{i} @{username} | style={style:.3f} | final={final:.3f}"
        gallery.append((img, caption))
    return gallery


def _save_upload(file: UploadFile) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    out = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with out.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return out


def _copy_visual_outputs(run_id: str) -> dict[str, str]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "global_pca": OUT_PLOT,
        "global_pca_3d": OUT_PLOT_3D,
        "local_topk_pca": OUT_LOCAL_PLOT,
        "topk_similarity": OUT_SIM_PLOT,
        "ranking_json": OUT_PLOT.with_suffix(".json"),
    }
    out_paths: dict[str, str] = {}
    for key, src in targets.items():
        if src.is_file():
            dst = run_dir / src.name
            shutil.copy2(src, dst)
            out_paths[key] = str(dst)
    return out_paths


def _profiles_payload() -> dict[str, Any]:
    if not INFLUENCERS_GEO_JSON.is_file():
        raise HTTPException(status_code=404, detail="Missing influencers_geo.json")

    geo = json.loads(INFLUENCERS_GEO_JSON.read_text(encoding="utf-8"))
    if not isinstance(geo, dict):
        raise HTTPException(status_code=400, detail="Invalid influencers_geo.json format")

    features = _load_features()
    style_profiles = {p["username"]: p for p in _load_profiles()}

    rows: list[dict[str, Any]] = []
    for username, geo_data in geo.items():
        u = str(username).lower().strip()
        feat = features.get(u, {})
        sp = style_profiles.get(u, {})
        top_hashtags = feat.get("top_hashtags") or []
        rows.append(
            {
                "username": u,
                "country": (geo_data or {}).get("country"),
                "city": (geo_data or {}).get("city"),
                "avg_likes": float(feat.get("avg_likes") or 0.0),
                "avg_comments": float(feat.get("avg_comments") or 0.0),
                "top_hashtags": top_hashtags,
                "style_vector_dim": int(len(sp.get("centroid", []))) if sp else None,
            }
        )
    rows.sort(key=lambda r: r["username"])
    return {"count": len(rows), "profiles": rows}


app = FastAPI(title="Scuffers Creator Match AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/profiles")
def profiles() -> dict[str, Any]:
    return _profiles_payload()


@app.post("/match")
async def match_endpoint(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    country: str | None = Form(None),
    city: str | None = Form(None),
    keywords: str | None = Form(None),
    w_style: float = Form(0.70),
    w_geo: float = Form(0.15),
    w_engagement: float = Form(0.10),
    w_topic: float = Form(0.05),
    country_mandatory: bool = Form(False),
    city_mandatory: bool = Form(False),
    min_style_for_business: float = Form(0.60),
    device: str = Form("cpu"),
) -> JSONResponse:
    path = _save_upload(image)
    try:
        payload = run_match(
            product_image=path,
            top_k=max(1, int(top_k)),
            country=country,
            city=city,
            keywords=_parse_keywords(keywords),
            w_style=float(w_style),
            w_geo=float(w_geo),
            w_engagement=float(w_engagement),
            w_topic=float(w_topic),
            device=device,
            country_mandatory=bool(country_mandatory),
            city_mandatory=bool(city_mandatory),
            min_style_for_business=float(min_style_for_business),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(payload)


@app.post("/match/style-only")
async def match_style_only_endpoint(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    country: str | None = Form(None),
    city: str | None = Form(None),
    keywords: str | None = Form(None),
    country_mandatory: bool = Form(False),
    city_mandatory: bool = Form(False),
    device: str = Form("cpu"),
) -> JSONResponse:
    path = _save_upload(image)
    try:
        payload = run_match(
            product_image=path,
            top_k=max(1, int(top_k)),
            country=country,
            city=city,
            keywords=_parse_keywords(keywords),
            w_style=1.0,
            w_geo=0.0,
            w_engagement=0.0,
            w_topic=0.0,
            device=device,
            country_mandatory=bool(country_mandatory),
            city_mandatory=bool(city_mandatory),
            min_style_for_business=0.0,
        )
        payload["rank_mode"] = "style-only"
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(payload)


@app.post("/match/business")
async def match_business_endpoint(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    country: str | None = Form(None),
    city: str | None = Form(None),
    keywords: str | None = Form(None),
    country_mandatory: bool = Form(False),
    city_mandatory: bool = Form(False),
    min_style_for_business: float = Form(0.60),
    device: str = Form("cpu"),
) -> JSONResponse:
    path = _save_upload(image)
    try:
        payload = run_match(
            product_image=path,
            top_k=max(1, int(top_k)),
            country=country,
            city=city,
            keywords=_parse_keywords(keywords),
            w_style=0.70,
            w_geo=0.15,
            w_engagement=0.10,
            w_topic=0.05,
            device=device,
            country_mandatory=bool(country_mandatory),
            city_mandatory=bool(city_mandatory),
            min_style_for_business=float(min_style_for_business),
        )
        payload["rank_mode"] = "business"
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(payload)


@app.post("/visualize")
async def visualize_endpoint(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    country: str | None = Form(None),
    city: str | None = Form(None),
    keywords: str | None = Form(None),
    w_style: float = Form(0.70),
    w_geo: float = Form(0.15),
    w_engagement: float = Form(0.10),
    w_topic: float = Form(0.05),
    country_mandatory: bool = Form(False),
    city_mandatory: bool = Form(False),
    min_style_for_business: float = Form(0.60),
    device: str = Form("cpu"),
) -> JSONResponse:
    path = _save_upload(image)
    run_id = uuid.uuid4().hex[:10]
    try:
        run_visualize(
            image_path=path,
            top_k=max(1, int(top_k)),
            country=country,
            city=city,
            keywords=_parse_keywords(keywords),
            w_style=float(w_style),
            w_geo=float(w_geo),
            w_engagement=float(w_engagement),
            w_topic=float(w_topic),
            device=device,
            country_mandatory=bool(country_mandatory),
            city_mandatory=bool(city_mandatory),
            min_style_for_business=float(min_style_for_business),
        )
        out_paths = _copy_visual_outputs(run_id)
        ranking_json = out_paths.get("ranking_json")
        ranking = (
            json.loads(Path(ranking_json).read_text(encoding="utf-8"))
            if ranking_json and Path(ranking_json).is_file()
            else None
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(
        {
            "run_id": run_id,
            "input_image": str(path),
            "outputs": out_paths,
            "ranking": ranking,
        }
    )


def _gradio_infer(
    image_path: str,
    rank_mode: str,
    city: str,
    country: str,
    city_mandatory: bool,
    country_mandatory: bool,
    keywords: str,
    w_style: float,
    w_geo: float,
    w_engagement: float,
    w_topic: float,
) -> tuple[str, str | None, str | None, list[tuple[str, str]]]:
    if not image_path:
        return "Upload an image first.", None, None, []

    mode = "style-only" if rank_mode == "Style only" else "business"
    w_style, w_geo, w_engagement, w_topic = _mode_weights(mode)

    run_visualize(
        image_path=Path(image_path),
        top_k=TOP_K_FIXED,
        country=country or None,
        city=city or None,
        keywords=_parse_keywords(keywords),
        w_style=w_style,
        w_geo=w_geo,
        w_engagement=w_engagement,
        w_topic=w_topic,
        device="cpu",
        country_mandatory=bool(country_mandatory),
        city_mandatory=bool(city_mandatory),
        min_style_for_business=0.60,
    )
    ranking_path = OUT_PLOT.with_suffix(".json")
    ranking = {}
    if ranking_path.is_file():
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["rank_mode"] = rank_mode
    ranking["top_k_fixed"] = TOP_K_FIXED
    summary = json.dumps(ranking, ensure_ascii=False, indent=2)
    gallery = _top_gallery_from_ranking(ranking)
    return summary, str(OUT_PLOT_3D), str(OUT_SIM_PLOT), gallery


with gr.Blocks(title="Scuffers Creator Match AI") as demo:
    gr.Markdown(
        "## Scuffers Creator Match AI\n"
        "Upload product image and get top influencer matches.\n\n"
        f"**Ranking mode**: Style only (pure visual) / Business rank (visual + business signals).\n"
        f"Top-K is fixed to **{TOP_K_FIXED}** for consistent demo output."
    )
    with gr.Row():
        rank_mode = gr.Radio(
            ["Style only", "Business rank"],
            value="Business rank",
            label="Ranking mode",
            info="Style only = pure visual similarity. Business rank = visual + geo + engagement + topic.",
        )
        city = gr.Textbox(label="City (optional)")
        country = gr.Textbox(label="Country (optional)")
    with gr.Row():
        city_mandatory = gr.Checkbox(value=True, label="City mandatory filter")
        country_mandatory = gr.Checkbox(value=True, label="Country mandatory filter")
    keywords = gr.Textbox(label="Keywords comma-separated", value="streetwear,oversized,hoodie")
    with gr.Row():
        in_image = gr.Image(type="filepath", label="Product image", height=260)
    # Hidden placeholders kept to preserve function signature and avoid UI clutter.
    w_style = gr.State(0.70)
    w_geo = gr.State(0.15)
    w_engagement = gr.State(0.10)
    w_topic = gr.State(0.05)
    run_btn = gr.Button("Run Match + Visualize", variant="primary")
    with gr.Tabs():
        with gr.Tab("3D Space"):
            out_global_3d = gr.Image(label="Global PCA 3D", height=360)
        with gr.Tab("Top-K Similarity"):
            out_sim = gr.Image(label="Top-K cosine similarity", height=300)
        with gr.Tab("Top Influencers"):
            out_gallery = gr.Gallery(
                label="Top influencer matches (username + representative photo)",
                columns=3,
                rows=1,
                height=260,
                object_fit="contain",
                preview=True,
            )
    with gr.Accordion("Ranking JSON (debug)", open=False):
        out_json = gr.Code(label="Ranking JSON", language="json", lines=12)

    run_btn.click(
        _gradio_infer,
        inputs=[
            in_image,
            rank_mode,
            city,
            country,
            city_mandatory,
            country_mandatory,
            keywords,
            w_style,
            w_geo,
            w_engagement,
            w_topic,
        ],
        outputs=[out_json, out_global_3d, out_sim, out_gallery],
    )


app = gr.mount_gradio_app(app, demo, path="/ui")


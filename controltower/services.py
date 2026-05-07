from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

from hackathon.scuffers_engine import run_engine


REPO_ROOT = Path(settings.BASE_DIR)
DATA_DIR = REPO_ROOT / "hackathon"
DEFAULT_CANDIDATE_ID = "#SCF-2026-6495"
CITY_COORDS = {
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3874, 2.1686),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Malaga": (36.7213, -4.4214),
    "Bilbao": (43.2630, -2.9350),
    "Zaragoza": (41.6488, -0.8891),
    "Palma": (39.5696, 2.6502),
    "Alicante": (38.3452, -0.4810),
    "Murcia": (37.9922, -1.1307),
}
_OPS_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}


def get_ops_context(candidate_id: str = DEFAULT_CANDIDATE_ID, fetch_shipping: bool = True) -> dict[str, Any]:
    now = time.time()
    if _OPS_CACHE["data"] is not None and (now - float(_OPS_CACHE["ts"])) < 45:
        return _OPS_CACHE["data"]

    result = run_engine(
        DATA_DIR,
        candidate_id=candidate_id or None,
        fetch_shipping=fetch_shipping,
        shipping_limit=8,
    )
    actions = result["actions"]
    sku_table = result["sku_table"].sort_values("sku_risk", ascending=False)
    city_issues = result["city_issues"]
    service_routes = result["service_routes"]
    shipping_statuses = result["shipping_statuses"]

    top_city = None
    if not city_issues.empty:
        city_summary = _city_summary(city_issues)
        top_city = city_summary.iloc[0].to_dict()
        city_summary["lat"] = city_summary["shipping_city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
        city_summary["lon"] = city_summary["shipping_city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
        city_summary = city_summary.dropna(subset=["lat", "lon"])
    else:
        city_summary = pd.DataFrame()

    payload = {
        "actions": actions,
        "top_actions": actions[:5],
        "top_city": top_city,
        "top_sku": sku_table.head(1).to_dict(orient="records")[0] if not sku_table.empty else None,
        "sku_rows": sku_table.head(12).to_dict(orient="records"),
        "city_rows": city_summary.head(10).to_dict(orient="records") if not city_summary.empty else [],
        "client_rows": _prioritized_clients(service_routes),
        "low_order_products": get_low_order_products(limit=10),
        "shipping_rows": shipping_statuses.head(10).to_dict(orient="records") if not shipping_statuses.empty else [],
        "shipping_queried": int(len(shipping_statuses)),
        "shipping_errors": int((shipping_statuses["shipping_api_status"] != "ok").sum()) if not shipping_statuses.empty else 0,
        "city_map_rows": city_summary.to_dict(orient="records") if not city_summary.empty else [],
    }
    _OPS_CACHE["ts"] = now
    _OPS_CACHE["data"] = payload
    return payload


def _contact_for_channel(channel: str, customer_id: str) -> tuple[str, str]:
    ch = str(channel or "").strip().lower()
    message = f"Hola {customer_id}, te escribimos de Scuffers para ayudarte con tu pedido."
    if ch == "instagram_dm":
        return "Escribir por Instagram DM", "https://www.instagram.com/direct/inbox/"
    if ch == "email":
        return "Enviar email", f"mailto:?subject=Scuffers%20support%20{customer_id}&body={message}"
    return "Abrir chat", f"https://wa.me/?text={message.replace(' ', '%20')}"


def _prioritized_clients(service_routes: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    if service_routes.empty:
        return []
    rows = (
        service_routes.sort_values("service_priority", ascending=False)
        .drop_duplicates(subset=["customer_id"])
        .head(limit)
        .copy()
    )
    rows["channel"] = rows["channel"].fillna("chat")
    out = []
    for _, row in rows.iterrows():
        label, url = _contact_for_channel(str(row.get("channel", "chat")), str(row.get("customer_id", "")))
        out.append(
            {
                "customer_id": row.get("customer_id"),
                "customer_segment": row.get("customer_segment"),
                "shipping_city": row.get("shipping_city"),
                "service_route": row.get("service_route"),
                "service_priority": round(float(row.get("service_priority") or 0), 2),
                "channel": row.get("channel"),
                "contact_label": label,
                "contact_url": url,
            }
        )
    return out


def _city_summary(city_issues: pd.DataFrame) -> pd.DataFrame:
    return (
        city_issues.groupby("shipping_city", as_index=False)
        .agg(
            total_tickets=("tickets", "sum"),
            operational_heat=("operational_heat", "max"),
            avg_city_urgency=("avg_city_urgency", "max"),
        )
        .sort_values(["total_tickets", "operational_heat"], ascending=[False, False])
    )


def get_low_order_products(limit: int = 10) -> list[dict[str, Any]]:
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").fillna(0)
    by_sku = orders.groupby("sku", as_index=False).agg(
        quantity_ordered=("quantity", "sum"),
        orders=("order_id", "count"),
    )
    products = (
        inventory[["sku", "product_name"]]
        .merge(by_sku, on="sku", how="left")
        .fillna({"quantity_ordered": 0, "orders": 0})
    )
    products = (
        products.groupby("product_name", as_index=False)
        .agg(
            quantity_ordered=("quantity_ordered", "sum"),
            orders=("orders", "sum"),
            sample_sku=("sku", "first"),
        )
        .sort_values(["quantity_ordered", "orders", "product_name"], ascending=True)
        .head(limit)
    )
    products["quantity_ordered"] = products["quantity_ordered"].astype(int)
    products["orders"] = products["orders"].astype(int)
    return products.to_dict(orient="records")


def save_uploaded_file(uploaded_file: Any) -> Path:
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name or "product.jpg").suffix or ".jpg"
    target = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return target


def _media_url(path: Path) -> str:
    rel = path.resolve().relative_to(Path(settings.MEDIA_ROOT).resolve())
    return settings.MEDIA_URL + str(rel).replace("\\", "/")


def _copy_to_media(src: Path, run_id: str) -> str | None:
    if not src.is_file():
        return None
    out_dir = Path(settings.MEDIA_ROOT) / "matches" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return _media_url(dst)


def _representative_image(username: str, run_id: str) -> str | None:
    folder = REPO_ROOT / "data" / "influencers" / username / "images"
    if not folder.is_dir():
        return None
    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return _copy_to_media(images[0], run_id) if images else None


def run_creator_match(
    uploaded_file: Any,
    mode: str,
    city: str,
    country: str,
    keywords: str,
) -> dict[str, Any]:
    from matching.match_product import run_match

    image_path = save_uploaded_file(uploaded_file)
    run_id = uuid.uuid4().hex[:10]
    is_style = mode == "style"
    weights = (1.0, 0.0, 0.0, 0.0) if is_style else (0.70, 0.15, 0.10, 0.05)
    keyword_list = [k.strip() for k in (keywords or "").split(",") if k.strip()]

    ranking = run_match(
        product_image=image_path,
        top_k=3,
        country=(country or "").strip() or None,
        city=(city or "").strip() or None,
        keywords=keyword_list,
        w_style=weights[0],
        w_geo=weights[1],
        w_engagement=weights[2],
        w_topic=weights[3],
        device="cpu",
        country_mandatory=False,
        city_mandatory=False,
        min_style_for_business=0.0 if is_style else 0.60,
    )

    top_rows = []
    for idx, row in enumerate(ranking.get("top_k", []), start=1):
        username = str(row.get("username") or "")
        top_rows.append(
            {
                "rank": idx,
                "username": username,
                "style_match_percent": row.get("style_match_percent"),
                "final_match_percent": row.get("final_match_percent"),
                "city": row.get("city"),
                "country": row.get("country"),
                "image": _representative_image(username, run_id),
            }
        )

    return {
        "input_image": _media_url(image_path),
        "top_influencers": top_rows,
        "global_plot": None,
        "similarity_plot": None,
        "ranking": ranking,
    }

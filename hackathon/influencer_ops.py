from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FALLBACK_CREATORS = [
    {"username": "madrid.streets", "city": "Madrid", "country": "ES", "avg_likes": 4200, "topics": ["streetwear", "hoodie", "drop"]},
    {"username": "barcelonafits", "city": "Barcelona", "country": "ES", "avg_likes": 3800, "topics": ["tshirt", "minimal", "outfit"]},
    {"username": "valencia.archive", "city": "Valencia", "country": "ES", "avg_likes": 2600, "topics": ["cap", "summer", "streetwear"]},
    {"username": "sevilla.dropclub", "city": "Sevilla", "country": "ES", "avg_likes": 2100, "topics": ["hoodie", "sneakers", "launch"]},
    {"username": "bilbao.uniform", "city": "Bilbao", "country": "ES", "avg_likes": 1900, "topics": ["knit", "minimal", "dailyfit"]},
    {"username": "malaga.streetnotes", "city": "Malaga", "country": "ES", "avg_likes": 2300, "topics": ["tee", "cap", "streetwear"]},
]
INFLUENCERS_DIR = REPO_ROOT / "data" / "influencers"
INFLUENCERS_GEO_JSON = INFLUENCERS_DIR / "influencers_geo.json"
FEATURES_DB = INFLUENCERS_DIR / "influencer_features.db"
FEATURES_JSON = INFLUENCERS_DIR / "influencer_features.json"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _load_creator_pool() -> List[Dict[str, Any]]:
    try:
        if not INFLUENCERS_GEO_JSON.is_file():
            raise FileNotFoundError("missing influencers_geo.json")
        features = _load_feature_map()
        geo = json.loads(INFLUENCERS_GEO_JSON.read_text(encoding="utf-8"))
        rows = []
        for username, geo_data in geo.items():
            feat = features.get(str(username).lower(), {})
            tags = [
                str(tag.get("tag", "")).lower().lstrip("#")
                for tag in (feat.get("top_hashtags") or [])
                if isinstance(tag, dict)
            ]
            rows.append(
                {
                    "username": str(username).lower(),
                    "city": (geo_data or {}).get("city"),
                    "country": (geo_data or {}).get("country"),
                    "avg_likes": float(feat.get("avg_likes") or 0),
                    "topics": tags,
                    "source": "creator_match_profiles",
                }
            )
        if rows:
            return rows
    except Exception:
        pass

    return [{**creator, "source": "fallback_demo_pool"} for creator in FALLBACK_CREATORS]


def _load_feature_map() -> Dict[str, Dict[str, Any]]:
    if FEATURES_DB.is_file():
        with sqlite3.connect(FEATURES_DB) as conn:
            rows = conn.execute(
                "SELECT username, avg_likes, avg_comments, top_hashtags_json FROM influencer_features"
            ).fetchall()
        out = {}
        for username, avg_likes, avg_comments, tags_json in rows:
            try:
                tags = json.loads(tags_json or "[]")
            except json.JSONDecodeError:
                tags = []
            out[str(username).lower()] = {
                "avg_likes": float(avg_likes or 0),
                "avg_comments": float(avg_comments or 0),
                "top_hashtags": tags,
            }
        return out

    if FEATURES_JSON.is_file():
        payload = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
        out = {}
        for row in payload.get("rows", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            username = str(row.get("username") or "").lower()
            if username:
                out[username] = row
        return out

    return {}


def _keyword_set(row: pd.Series) -> set[str]:
    words = {
        str(row.get("category") or "").lower(),
        str(row.get("product_name") or "").lower(),
        str(row.get("sku") or "").lower(),
    }
    expanded = set()
    for word in words:
        expanded.update(part for part in word.replace("-", " ").split() if part)
    if "tshirt" in expanded or "tee" in expanded:
        expanded.update(["tee", "tshirt"])
    if "hoodie" in expanded:
        expanded.update(["hoodie", "streetwear"])
    if "cap" in expanded:
        expanded.update(["cap", "accessories"])
    return expanded


def _rank_creators(creators: List[Dict[str, Any]], city: str, keywords: set[str], top_k: int = 3) -> List[Dict[str, Any]]:
    ranked = []
    target_city = str(city or "").strip().lower()
    for creator in creators:
        creator_city = str(creator.get("city") or "").strip().lower()
        topics = {str(topic).lower().lstrip("#") for topic in (creator.get("topics") or [])}
        geo_score = 1.0 if target_city and creator_city == target_city else 0.35
        topic_score = min(1.0, len(topics.intersection(keywords)) / max(1, min(4, len(keywords))))
        engagement_score = min(1.0, float(creator.get("avg_likes") or 0) / 5000)
        final_score = 0.48 * geo_score + 0.32 * topic_score + 0.20 * engagement_score
        ranked.append(
            {
                "username": creator["username"],
                "city": creator.get("city"),
                "avg_likes": round(float(creator.get("avg_likes") or 0), 0),
                "score": round(final_score, 3),
                "source": creator.get("source", "unknown"),
            }
        )
    return sorted(ranked, key=lambda r: r["score"], reverse=True)[:top_k]


def build_influencer_recommendations(data_dir: Path, max_rows: int = 10) -> pd.DataFrame:
    orders = _safe_read_csv(data_dir / "orders.csv")
    inventory = _safe_read_csv(data_dir / "inventory.csv")
    campaigns = _safe_read_csv(data_dir / "campaigns.csv")
    if orders.empty or inventory.empty:
        return pd.DataFrame()

    creators = _load_creator_pool()
    sku_city = orders.groupby(["sku", "shipping_city"], as_index=False).agg(orders=("order_id", "count"))
    city_totals = orders.groupby("shipping_city", as_index=False).agg(city_orders=("order_id", "count"))
    sku_totals = orders.groupby("sku", as_index=False).agg(sku_orders=("order_id", "count"))
    base = inventory.merge(sku_totals, on="sku", how="left")
    base["sku_orders"] = pd.to_numeric(base["sku_orders"], errors="coerce").fillna(0)
    base["product_page_views_last_hour"] = pd.to_numeric(base["product_page_views_last_hour"], errors="coerce").fillna(0)
    base["conversion_rate_last_hour"] = pd.to_numeric(base["conversion_rate_last_hour"], errors="coerce").fillna(0)
    base["under_sales_score"] = (
        (base["product_page_views_last_hour"] * base["conversion_rate_last_hour"]) / (base["sku_orders"] + 1)
    ).round(2)

    rows = []
    cities = city_totals.sort_values("city_orders", ascending=True)["shipping_city"].head(4).tolist()
    campaigns_by_sku = campaigns.groupby("target_sku", as_index=False).agg(active_campaigns=("campaign_id", "count"))
    base = base.merge(campaigns_by_sku, left_on="sku", right_on="target_sku", how="left")
    base["active_campaigns"] = base["active_campaigns"].fillna(0)

    for _, sku_row in base.sort_values("under_sales_score", ascending=False).head(6).iterrows():
        sku = str(sku_row["sku"])
        keywords = _keyword_set(sku_row)
        for city in cities:
            current_orders = sku_city[(sku_city["sku"] == sku) & (sku_city["shipping_city"] == city)]["orders"]
            city_orders = int(current_orders.iloc[0]) if not current_orders.empty else 0
            if city_orders > 2:
                continue
            creators_ranked = _rank_creators(creators, city, keywords)
            rows.append(
                {
                    "sku": sku,
                    "product_name": sku_row.get("product_name"),
                    "city": city,
                    "orders_in_city": city_orders,
                    "under_sales_score": float(sku_row["under_sales_score"]),
                    "recommended_creators": ", ".join([f"@{c['username']}" for c in creators_ranked]),
                    "creator_source": creators_ranked[0]["source"] if creators_ranked else "none",
                    "brief": (
                        f"Subir una foto/reel del producto {sku_row.get('product_name')} para {city}: "
                        f"styling rápido, urgencia de drop limitado y CTA a stock disponible."
                    ),
                    "why": (
                        f"Pocas ventas locales ({city_orders}) pese a señales de demanda del SKU "
                        f"({float(sku_row['under_sales_score']):.2f})."
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(["under_sales_score", "orders_in_city"], ascending=[False, True]).head(max_rows)

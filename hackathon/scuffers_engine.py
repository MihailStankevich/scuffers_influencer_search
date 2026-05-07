import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import requests

    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


SHIPPING_STATUS_URL = "https://lkuutmnykcnbfmbpopcu.functions.supabase.co/api/shipping-status/{order_id}"
SHIPPING_STATUS_RISK = {
    "label_created": 0.35,
    "picked_up": 0.20,
    "in_transit": 0.15,
    "at_sorting_center": 0.25,
    "out_for_delivery": 0.10,
    "delivered": 0.00,
    "delayed": 0.85,
    "exception": 1.00,
    "lost": 1.00,
    "returned_to_sender": 0.95,
}


@dataclass
class Action:
    action_type: str
    target_id: str
    title: str
    reason: str
    expected_impact: str
    confidence: float
    owner: str
    automation_possible: bool
    importance_score: float
    risk_reduction: float
    customer_impact: float
    operational_cost: float


def _build_openai_explanation(action: Action) -> str:
    """
    Optional LLM explanation layer.
    Uses OpenAI only if OPENAI_API_KEY is present and openai package is installed.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return explain_action(action)
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return explain_action(action)

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Eres un Operations Lead de Scuffers. Explica en 2 frases concretas y accionables "
            "por qué esta acción es prioritaria durante un lanzamiento de alta demanda.\n"
            f"action_type={action.action_type}\n"
            f"title={action.title}\n"
            f"reason={action.reason}\n"
            f"expected_impact={action.expected_impact}\n"
            f"importance={action.importance_score:.2f}\n"
            f"risk_reduction={action.risk_reduction:.2f}\n"
            f"customer_impact={action.customer_impact:.2f}\n"
            f"operational_cost={action.operational_cost:.2f}\n"
        )
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=140,
        )
        text = (resp.output_text or "").strip()
        return text if text else explain_action(action)
    except Exception:
        return explain_action(action)


def _normalize_order_value(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": None})
        .astype(float)
    )


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    orders = pd.read_csv(data_dir / "orders.csv")
    order_items = pd.read_csv(data_dir / "order_items.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    inventory = pd.read_csv(data_dir / "inventory.csv")
    tickets = pd.read_csv(data_dir / "support_tickets.csv")
    campaigns = pd.read_csv(data_dir / "campaigns.csv")

    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True)
    orders["order_value_clean"] = _normalize_order_value(orders["order_value"])
    tickets["created_at"] = pd.to_datetime(tickets["created_at"], errors="coerce", utc=True)

    for col in [
        "inventory_available_units",
        "inventory_reserved_units",
        "inventory_incoming_units",
        "product_page_views_last_hour",
        "conversion_rate_last_hour",
        "sell_through_rate_last_hour",
    ]:
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce").fillna(0)

    campaigns["traffic_growth"] = pd.to_numeric(campaigns["traffic_growth"], errors="coerce").fillna(0)
    campaigns["conversion_rate"] = pd.to_numeric(campaigns["conversion_rate"], errors="coerce").fillna(0)
    campaigns["campaign_intensity"] = campaigns["campaign_intensity"].fillna("low")

    return {
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
        "inventory": inventory,
        "tickets": tickets,
        "campaigns": campaigns,
    }


def _eta_hours(eta_series: pd.Series, reference_ts: pd.Timestamp) -> pd.Series:
    eta = pd.to_datetime(eta_series, errors="coerce", utc=True)
    return ((eta - reference_ts).dt.total_seconds() / 3600).clip(lower=0).fillna(48)


def build_sku_risk_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"]
    inventory = data["inventory"].copy()
    tickets = data["tickets"].copy()

    urgency_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    tickets["urgency_score"] = tickets["support_ticket_urgency"].str.lower().map(urgency_map).fillna(1)

    orders_for_tickets = orders[["order_id", "sku"]].drop_duplicates()
    tickets_sku = tickets.merge(orders_for_tickets, on="order_id", how="left")
    ticket_agg = tickets_sku.groupby("sku", as_index=False).agg(
        ticket_count=("ticket_id", "count"),
        avg_ticket_urgency=("urgency_score", "mean"),
    )

    order_agg = orders.groupby("sku", as_index=False).agg(
        sku_orders=("order_id", "count"),
        express_orders=("shipping_method", lambda s: s.astype(str).str.lower().eq("express").sum()),
    )

    inv = inventory.merge(ticket_agg, on="sku", how="left").merge(order_agg, on="sku", how="left")
    inv["ticket_count"] = inv["ticket_count"].fillna(0)
    inv["avg_ticket_urgency"] = inv["avg_ticket_urgency"].fillna(1)
    inv["sku_orders"] = inv["sku_orders"].fillna(0)
    inv["express_orders"] = inv["express_orders"].fillna(0)

    ref_ts = orders["created_at"].max()
    inv["eta_hours"] = _eta_hours(inv["inventory_incoming_eta"], ref_ts)
    inv["stock_stress"] = inv["inventory_reserved_units"] / (inv["inventory_available_units"] + 1)
    inv["demand_stress"] = (
        inv["product_page_views_last_hour"] * inv["conversion_rate_last_hour"]
    ) / (inv["inventory_available_units"] + 1)
    inv["incoming_relief"] = inv["inventory_incoming_units"] / (inv["eta_hours"] + 1)
    inv["ticket_pressure"] = inv["ticket_count"] / (inv["sku_orders"] + 1)
    inv["express_pressure"] = inv["express_orders"] / (inv["sku_orders"] + 1)

    raw = (
        0.33 * inv["stock_stress"].clip(upper=20)
        + 0.25 * inv["demand_stress"].clip(upper=20)
        + 0.20 * (inv["ticket_pressure"] * inv["avg_ticket_urgency"]).clip(upper=10)
        + 0.14 * inv["express_pressure"].clip(upper=5)
        - 0.20 * inv["incoming_relief"].clip(upper=5)
    )
    inv["sku_risk"] = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 10).round(2)
    return inv


def _candidate_order_pressure(data: Dict[str, pd.DataFrame], sku_table: pd.DataFrame) -> pd.DataFrame:
    orders = data["orders"]
    customers = data["customers"]
    tickets = data["tickets"]
    ticket_orders = tickets[["order_id", "support_ticket_urgency"]].copy()
    base = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(sku_table[["sku", "sku_risk", "inventory_available_units"]], on="sku", how="left")
        .merge(ticket_orders, on="order_id", how="left")
    )
    base["ticket_urgency_score"] = (
        base["support_ticket_urgency"].str.lower().map({"low": 1, "medium": 2, "high": 3, "urgent": 4}).fillna(0)
    )
    base["pre_api_order_pressure"] = (
        0.45 * base["sku_risk"].fillna(0)
        + 0.20 * base["ticket_urgency_score"]
        + 0.20 * base["shipping_method"].astype(str).str.lower().eq("express").astype(int) * 4
        + 0.15 * base["is_vip"].fillna(False).astype(int) * 5
    )
    return base


def _fetch_shipping_status(order_id: str, candidate_id: str, timeout: float = 1.2) -> Dict[str, Any]:
    if not HAS_REQUESTS:
        return {"order_id": order_id, "shipping_api_status": "error", "shipping_api_error": "requests_not_installed"}
    try:
        response = requests.get(
            SHIPPING_STATUS_URL.format(order_id=order_id),
            headers={"X-Candidate-Id": candidate_id.strip().lstrip("#")},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        if not isinstance(payload, dict):
            raise ValueError("Shipping API returned a non-object payload")
        payload["order_id"] = str(payload.get("order_id") or order_id)
        payload["shipping_api_status"] = "ok"
        return payload
    except Exception as exc:
        return {"order_id": order_id, "shipping_api_status": "error", "shipping_api_error": str(exc)[:180]}


def fetch_relevant_shipping_statuses(
    data: Dict[str, pd.DataFrame],
    sku_table: pd.DataFrame,
    candidate_id: Optional[str] = None,
    limit: int = 20,
) -> pd.DataFrame:
    columns = [
        "order_id",
        "shipping_status",
        "delay_risk",
        "delay_reason",
        "estimated_delivery_date",
        "requires_manual_review",
        "delivery_attempts",
        "shipping_api_status",
        "shipping_api_error",
        "shipping_status_risk",
        "shipping_api_risk",
        "pre_api_order_pressure",
    ]
    if not candidate_id:
        return pd.DataFrame(columns=columns)

    candidates = _candidate_order_pressure(data, sku_table).sort_values("pre_api_order_pressure", ascending=False)
    candidate_rows = list(candidates.head(max(1, int(limit))).iterrows())
    if not candidate_rows:
        return pd.DataFrame(columns=columns)

    records = []
    with ThreadPoolExecutor(max_workers=min(8, len(candidate_rows))) as pool:
        futures = {
            pool.submit(_fetch_shipping_status, str(row["order_id"]), candidate_id): float(row["pre_api_order_pressure"])
            for _, row in candidate_rows
        }
        for future in as_completed(futures):
            payload = future.result()
            payload["pre_api_order_pressure"] = futures[future]
            records.append(payload)

    status_df = pd.DataFrame(records)
    for col in columns:
        if col not in status_df.columns:
            status_df[col] = None
    status_df["delay_risk"] = pd.to_numeric(status_df["delay_risk"], errors="coerce").fillna(0).clip(0, 1)
    status_df["requires_manual_review"] = status_df["requires_manual_review"].fillna(False).astype(bool)
    status_df["shipping_status_risk"] = (
        status_df["shipping_status"].astype(str).str.lower().map(SHIPPING_STATUS_RISK).fillna(0.40)
    )
    status_df.loc[status_df["shipping_api_status"] != "ok", "shipping_status_risk"] = 0
    status_df["shipping_api_risk"] = (
        10
        * (
            0.55 * status_df["delay_risk"]
            + 0.30 * status_df["shipping_status_risk"]
            + 0.15 * status_df["requires_manual_review"].astype(float)
        )
    ).round(2)
    status_df.loc[status_df["shipping_api_status"] != "ok", "shipping_api_risk"] = 0
    return status_df[columns].sort_values("shipping_api_risk", ascending=False)


def build_correlations(sku_table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sku_risk",
        "ticket_pressure",
        "stock_stress",
        "demand_stress",
        "express_pressure",
        "incoming_relief",
        "sell_through_rate_last_hour",
    ]
    corr = sku_table[cols].corr(numeric_only=True)["sku_risk"].drop("sku_risk").sort_values(ascending=False)
    return corr.reset_index().rename(columns={"index": "feature", "sku_risk": "correlation_with_sku_risk"})


def _classify_ticket_issue(message: str) -> str:
    text = str(message).lower()
    if "cambiar la dirección" in text or "cambio la dirección" in text or "cambiar direccion" in text:
        return "cambio_direccion"
    if "no veo movimiento" in text or "sin movimiento" in text:
        return "no_movimiento_envio"
    if "confirmación de entrega" in text or "confirmacion de entrega" in text:
        return "confirmacion_entrega"
    if "llegará antes" in text or "llegara antes" in text:
        return "confirmacion_plazo"
    if "prioridad" in text or "pagado express" in text:
        return "solicitud_prioridad"
    if "se agote" in text:
        return "miedo_rotura_stock"
    return "otros"


def build_city_issue_table(data: Dict[str, pd.DataFrame], sku_table: pd.DataFrame) -> pd.DataFrame:
    orders = data["orders"]
    tickets = data["tickets"].copy()
    urgency_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    tickets["urgency_score"] = tickets["support_ticket_urgency"].str.lower().map(urgency_map).fillna(1)
    tickets["issue_type"] = tickets["support_ticket_message"].apply(_classify_ticket_issue)

    order_cols = orders[["order_id", "shipping_city", "sku"]].drop_duplicates()
    merged = tickets.merge(order_cols, on="order_id", how="left").merge(
        sku_table[["sku", "sku_risk"]], on="sku", how="left"
    )
    merged["sku_risk"] = merged["sku_risk"].fillna(0)

    city_issue = (
        merged.groupby(["shipping_city", "issue_type"], as_index=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_urgency=("urgency_score", "mean"),
            avg_sku_risk=("sku_risk", "mean"),
        )
    )
    city_total = (
        merged.groupby("shipping_city", as_index=False)
        .agg(
            total_tickets=("ticket_id", "count"),
            avg_city_urgency=("urgency_score", "mean"),
            avg_city_sku_risk=("sku_risk", "mean"),
        )
    )
    city_total["operational_heat"] = (
        0.45 * city_total["total_tickets"].clip(upper=30)
        + 0.30 * city_total["avg_city_urgency"]
        + 0.25 * city_total["avg_city_sku_risk"]
    )
    city_issue = city_issue.merge(city_total, on="shipping_city", how="left")
    return city_issue.sort_values(["operational_heat", "tickets"], ascending=False)


def build_customer_service_routes(
    data: Dict[str, pd.DataFrame],
    sku_table: pd.DataFrame,
    shipping_statuses: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    orders = data["orders"]
    customers = data["customers"]
    tickets = data["tickets"]
    urgency_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    sentiment_map = {"positive": 1, "neutral": 2, "negative": 3}

    t = tickets.copy()
    t["urgency_score"] = t["support_ticket_urgency"].str.lower().map(urgency_map).fillna(1)
    t["sentiment_score"] = t["support_ticket_sentiment"].str.lower().map(sentiment_map).fillna(2)

    base = (
        t.merge(orders[["order_id", "sku", "shipping_city", "shipping_method"]], on="order_id", how="left")
        .merge(
            customers[
                [
                    "customer_id",
                    "customer_segment",
                    "customer_lifetime_value",
                    "customer_orders_count",
                    "is_vip",
                    "email_opt_in",
                ]
            ],
            on="customer_id",
            how="left",
        )
        .merge(sku_table[["sku", "sku_risk"]], on="sku", how="left")
    )
    if shipping_statuses is not None and not shipping_statuses.empty:
        base = base.merge(
            shipping_statuses[
                [
                    "order_id",
                    "shipping_status",
                    "delay_risk",
                    "delay_reason",
                    "requires_manual_review",
                    "shipping_api_risk",
                ]
            ],
            on="order_id",
            how="left",
        )
    else:
        base["shipping_api_risk"] = 0
        base["delay_risk"] = 0
        base["shipping_status"] = None
        base["delay_reason"] = None
        base["requires_manual_review"] = False
    base["customer_lifetime_value"] = pd.to_numeric(base["customer_lifetime_value"], errors="coerce").fillna(0)
    base["customer_orders_count"] = pd.to_numeric(base["customer_orders_count"], errors="coerce").fillna(0)
    base["is_vip"] = base["is_vip"].fillna(False)
    base["email_opt_in"] = base["email_opt_in"].fillna(False)
    base["sku_risk"] = base["sku_risk"].fillna(0)
    base["shipping_api_risk"] = pd.to_numeric(base["shipping_api_risk"], errors="coerce").fillna(0)
    base["delay_risk"] = pd.to_numeric(base["delay_risk"], errors="coerce").fillna(0)
    base["requires_manual_review"] = base["requires_manual_review"].astype("boolean").fillna(False).astype(bool)

    base["service_route"] = "chatbot_first"
    high_value_mask = (base["is_vip"]) | (base["customer_lifetime_value"] >= 600) | (base["customer_orders_count"] >= 8)
    critical_mask = (base["urgency_score"] >= 3) | (base["sku_risk"] >= 7.5) | (base["shipping_api_risk"] >= 6)
    base.loc[high_value_mask & critical_mask, "service_route"] = "white_glove_human"
    base.loc[high_value_mask & ~critical_mask, "service_route"] = "human_priority"

    base["service_priority"] = (
        0.35 * base["urgency_score"]
        + 0.25 * base["sentiment_score"]
        + 0.18 * (base["sku_risk"] / 2.0)
        + 0.07 * base["shipping_api_risk"]
        + 0.15 * (base["customer_lifetime_value"] / 300).clip(upper=4)
    ).round(2)
    return base.sort_values("service_priority", ascending=False)


def explain_action(action: Action) -> str:
    return (
        f"{action.title}. Prioridad {action.importance_score:.2f}/10 porque reduce riesgo {action.risk_reduction:.2f}, "
        f"protege cliente {action.customer_impact:.2f} y tiene coste operativo {action.operational_cost:.2f}. "
        f"Impacto esperado: {action.expected_impact}"
    )


def generate_actions(
    data: Dict[str, pd.DataFrame],
    sku_table: pd.DataFrame,
    shipping_statuses: Optional[pd.DataFrame] = None,
) -> List[Action]:
    orders = data["orders"]
    customers = data["customers"]
    campaigns = data["campaigns"]
    tickets = data["tickets"]
    actions: List[Action] = []
    service_routes = build_customer_service_routes(data, sku_table, shipping_statuses)

    orders_enriched = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(sku_table[["sku", "sku_risk", "inventory_available_units"]], on="sku", how="left")
    )
    ticket_orders = tickets[["order_id", "support_ticket_urgency"]].copy()
    orders_enriched = orders_enriched.merge(ticket_orders, on="order_id", how="left")
    orders_enriched["ticket_urgency_score"] = (
        orders_enriched["support_ticket_urgency"].str.lower().map({"low": 1, "medium": 2, "high": 3, "urgent": 4}).fillna(0)
    )
    if shipping_statuses is not None and not shipping_statuses.empty:
        orders_enriched = orders_enriched.merge(
            shipping_statuses[
                [
                    "order_id",
                    "shipping_status",
                    "delay_risk",
                    "delay_reason",
                    "requires_manual_review",
                    "shipping_api_risk",
                    "shipping_api_status",
                ]
            ],
            on="order_id",
            how="left",
        )
    else:
        orders_enriched["shipping_api_risk"] = 0
        orders_enriched["delay_risk"] = 0
        orders_enriched["shipping_status"] = None
        orders_enriched["delay_reason"] = None
        orders_enriched["requires_manual_review"] = False
        orders_enriched["shipping_api_status"] = "not_requested"
    orders_enriched["shipping_api_risk"] = pd.to_numeric(orders_enriched["shipping_api_risk"], errors="coerce").fillna(0)
    orders_enriched["delay_risk"] = pd.to_numeric(orders_enriched["delay_risk"], errors="coerce").fillna(0)
    orders_enriched["requires_manual_review"] = (
        orders_enriched["requires_manual_review"].astype("boolean").fillna(False).astype(bool)
    )

    intensity_map = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
    campaign_enriched = campaigns.merge(
        sku_table[["sku", "sku_risk", "inventory_available_units", "inventory_incoming_units"]],
        left_on="target_sku",
        right_on="sku",
        how="left",
    )
    campaign_enriched["intensity_score"] = campaign_enriched["campaign_intensity"].str.lower().map(intensity_map).fillna(1)
    campaign_enriched["campaign_pressure"] = (
        0.5 * campaign_enriched["intensity_score"]
        + 0.3 * campaign_enriched["traffic_growth"]
        + 0.2 * campaign_enriched["conversion_rate"] * 10
    )
    campaign_enriched["action_pressure"] = campaign_enriched["campaign_pressure"] + campaign_enriched["sku_risk"].fillna(0)

    for _, row in campaign_enriched.sort_values("action_pressure", ascending=False).head(5).iterrows():
        pressure = float(row["action_pressure"])
        available = int(row.get("inventory_available_units", 0) or 0)
        if pressure >= 9 and available <= 5:
            a_type = "pause_campaign"
            impact = "Corta presión comercial sobre SKU crítico y evita oversell inmediato."
            op_cost = 1.8
        elif pressure >= 7:
            a_type = "reduce_campaign_intensity"
            impact = "Reduce demanda en pico sin apagar totalmente la adquisición."
            op_cost = 1.3
        else:
            a_type = "redirect_campaign"
            impact = "Mantiene conversiones desviando demanda a SKUs de menor riesgo."
            op_cost = 1.1
        importance = min(10.0, pressure)
        actions.append(
            Action(
                action_type=a_type,
                target_id=str(row["campaign_id"]),
                title=f"Ajustar campaña {row['campaign_id']} ({row['target_sku']})",
                reason=(
                    f"Presión de campaña {pressure:.2f} con SKU risk {float(row['sku_risk']):.2f} "
                    f"y stock disponible {available}."
                ),
                expected_impact=impact,
                confidence=round(min(0.95, 0.55 + pressure / 15), 2),
                owner="growth_ops",
                automation_possible=True,
                importance_score=importance,
                risk_reduction=min(10, pressure * 0.85),
                customer_impact=6.5,
                operational_cost=op_cost,
            )
        )

    orders_enriched["order_pressure"] = (
        0.34 * orders_enriched["sku_risk"].fillna(0)
        + 0.16 * orders_enriched["ticket_urgency_score"]
        + 0.16 * orders_enriched["shipping_method"].astype(str).str.lower().eq("express").astype(int) * 4
        + 0.12 * orders_enriched["is_vip"].fillna(False).astype(int) * 5
        + 0.22 * orders_enriched["shipping_api_risk"]
    )
    for _, row in orders_enriched.sort_values("order_pressure", ascending=False).head(8).iterrows():
        importance = min(10.0, float(row["order_pressure"]))
        api_reason = ""
        if float(row.get("shipping_api_risk", 0) or 0) > 0:
            api_reason = (
                f" API logística: estado {row.get('shipping_status')}, delay_risk "
                f"{float(row.get('delay_risk', 0)):.2f}, motivo {row.get('delay_reason') or 'n/a'}."
            )
        action_type = "manual_review_order" if bool(row.get("requires_manual_review", False)) else (
            "prioritize_order" if row.get("ticket_urgency_score", 0) >= 3 else "manual_review_order"
        )
        actions.append(
            Action(
                action_type=action_type,
                target_id=str(row["order_id"]),
                title=f"Intervenir pedido {row['order_id']} ({row['sku']})",
                reason=(
                    f"Presión pedido {importance:.2f} por SKU risk {float(row['sku_risk']):.2f}, "
                    f"urgencia ticket {int(row.get('ticket_urgency_score', 0))}, método {row['shipping_method']}."
                    f"{api_reason}"
                ),
                expected_impact="Reduce riesgo de incidencia logística y de mala experiencia del cliente.",
                confidence=round(min(0.95, 0.50 + importance / 15 + (0.05 if api_reason else 0)), 2),
                owner="operations",
                automation_possible=not bool(row.get("requires_manual_review", False)),
                importance_score=importance,
                risk_reduction=importance * 0.8,
                customer_impact=7.2 if bool(row.get("is_vip", False)) else 6.0,
                operational_cost=2.2,
            )
        )

    contact_candidates = orders_enriched[
        (orders_enriched["ticket_urgency_score"] >= 3) | (orders_enriched["delay_risk"] >= 0.60)
    ].sort_values(["delay_risk", "ticket_urgency_score"], ascending=False)
    for _, row in contact_candidates.head(4).iterrows():
        delay_context = ""
        if float(row.get("delay_risk", 0) or 0) >= 0.60:
            delay_context = (
                f" La API logística añade riesgo de retraso {float(row.get('delay_risk', 0)):.2f} "
                f"por {row.get('delay_reason') or 'motivo desconocido'}."
            )
        actions.append(
            Action(
                action_type="contact_customer_proactively",
                target_id=str(row["customer_id"]),
                title=f"Contactar cliente {row['customer_id']} antes de escalado",
                reason=(
                    f"Pedido {row['order_id']} con urgencia ticket alta y riesgo SKU "
                    f"{float(row['sku_risk']):.2f}.{delay_context}"
                ),
                expected_impact="Disminuye fricción y evita nuevos tickets de seguimiento.",
                confidence=0.84 if delay_context else 0.79,
                owner="support",
                automation_possible=True,
                importance_score=7.2 if delay_context else 6.8,
                risk_reduction=5.8 if delay_context else 5.2,
                customer_impact=8.6,
                operational_cost=1.0,
            )
        )

    # Service routing actions: protect high-value clients and offload new ones to chatbot.
    hv_cases = service_routes[service_routes["service_route"] == "white_glove_human"].head(3)
    for _, row in hv_cases.iterrows():
        actions.append(
            Action(
                action_type="assign_white_glove_agent",
                target_id=str(row["customer_id"]),
                title=f"Asignar agente senior a {row['customer_id']}",
                reason=(
                    f"Cliente de alto valor (CLV {float(row['customer_lifetime_value']):.0f}, "
                    f"VIP {bool(row['is_vip'])}) con incidencia sensible y prioridad {float(row['service_priority']):.2f}."
                ),
                expected_impact="Protege relación con cliente histórico y evita churn reputacional.",
                confidence=0.88,
                owner="support_lead",
                automation_possible=False,
                importance_score=min(10.0, 6.8 + float(row["service_priority"]) / 2),
                risk_reduction=min(10.0, 5.5 + float(row["service_priority"]) / 2.5),
                customer_impact=9.4,
                operational_cost=2.8,
            )
        )

    chatbot_cases = service_routes[service_routes["service_route"] == "chatbot_first"].head(3)
    for _, row in chatbot_cases.iterrows():
        actions.append(
            Action(
                action_type="route_to_chatbot",
                target_id=str(row["ticket_id"]),
                title=f"Derivar ticket {row['ticket_id']} a chatbot con playbook",
                reason=(
                    f"Cliente nuevo/bajo histórico con consulta tipo '{_classify_ticket_issue(row['support_ticket_message'])}' "
                    f"y prioridad {float(row['service_priority']):.2f}."
                ),
                expected_impact="Absorbe volumen repetitivo y libera agentes humanos para casos críticos.",
                confidence=0.84,
                owner="support_ops",
                automation_possible=True,
                importance_score=6.2,
                risk_reduction=4.8,
                customer_impact=5.9,
                operational_cost=0.7,
            )
        )

    return actions


def select_top_actions(actions: List[Action], max_actions: int = 10) -> List[Action]:
    selected: List[Action] = []
    used: Dict[str, int] = {}
    sorted_actions = sorted(
        actions,
        key=lambda a: (0.6 * a.importance_score + 0.25 * a.risk_reduction + 0.15 * a.customer_impact - 0.10 * a.operational_cost),
        reverse=True,
    )
    for action in sorted_actions:
        if len(selected) >= max_actions:
            break
        if used.get(action.action_type, 0) >= 3:
            continue
        if action.action_type == "pause_campaign" and used.get("pause_campaign", 0) >= 2:
            continue
        selected.append(action)
        used[action.action_type] = used.get(action.action_type, 0) + 1
    return selected


def actions_to_payload(actions: List[Action]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(actions, start=1):
        out.append(
            {
                "rank": i,
                "action_type": a.action_type,
                "target_id": a.target_id,
                "title": a.title,
                "reason": a.reason,
                "expected_impact": a.expected_impact,
                "confidence": round(a.confidence, 2),
                "owner": a.owner,
                "automation_possible": a.automation_possible,
                "importance_score": round(a.importance_score, 2),
                "explanation": _build_openai_explanation(a),
            }
        )
    return out


def run_engine(
    data_dir: Path,
    candidate_id: Optional[str] = None,
    fetch_shipping: bool = True,
    shipping_limit: int = 20,
) -> Dict[str, Any]:
    data = load_data(data_dir)
    sku_table = build_sku_risk_table(data)
    shipping_statuses = (
        fetch_relevant_shipping_statuses(data, sku_table, candidate_id=candidate_id, limit=shipping_limit)
        if fetch_shipping
        else pd.DataFrame()
    )
    correlations = build_correlations(sku_table)
    city_issues = build_city_issue_table(data, sku_table)
    service_routes = build_customer_service_routes(data, sku_table, shipping_statuses)
    actions = generate_actions(data, sku_table, shipping_statuses)
    top_actions = select_top_actions(actions, max_actions=10)
    payload = actions_to_payload(top_actions)
    return {
        "actions": payload,
        "sku_table": sku_table,
        "correlations": correlations,
        "city_issues": city_issues,
        "service_routes": service_routes,
        "shipping_statuses": shipping_statuses,
    }


def main() -> None:
    result = run_engine(Path(__file__).parent)
    print(json.dumps(result["actions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

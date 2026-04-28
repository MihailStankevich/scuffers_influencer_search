import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).parent


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
    impact_score: float
    effort_score: float
    revenue_penalty: float
    risk_reduction: float
    customer_protection: float
    operational_cost: float

    @property
    def total_score(self) -> float:
        # Multi-objective utility: maximize risk reduction and customer protection,
        # while avoiding high effort and unnecessary revenue damage.
        return (
            0.45 * self.risk_reduction
            + 0.35 * self.customer_protection
            + 0.20 * self.impact_score
            - 0.15 * self.operational_cost
            - 0.10 * self.revenue_penalty
            - 0.05 * self.effort_score
        )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_order_value(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": None})
        .astype(float)
    )


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    orders = _safe_read_csv(data_dir / "orders.csv")
    order_items = _safe_read_csv(data_dir / "order_items.csv")
    customers = _safe_read_csv(data_dir / "customers.csv")
    inventory = _safe_read_csv(data_dir / "inventory.csv")
    tickets = _safe_read_csv(data_dir / "support_tickets.csv")
    campaigns = _safe_read_csv(data_dir / "campaigns.csv")

    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True)
    orders["order_value_clean"] = _normalize_order_value(orders["order_value"])

    if "created_at" in tickets.columns:
        tickets["created_at"] = pd.to_datetime(tickets["created_at"], errors="coerce", utc=True)

    for col in ["inventory_available_units", "inventory_reserved_units", "inventory_incoming_units"]:
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce").fillna(0)

    campaigns["campaign_intensity"] = campaigns["campaign_intensity"].fillna("low")

    return {
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
        "inventory": inventory,
        "tickets": tickets,
        "campaigns": campaigns,
    }


def build_signals(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    orders = data["orders"]
    customers = data["customers"]
    inventory = data["inventory"]
    tickets = data["tickets"]
    campaigns = data["campaigns"]

    urgency_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    sentiment_map = {"positive": 1, "neutral": 2, "negative": 3}

    tickets = tickets.copy()
    tickets["urgency_score"] = tickets["support_ticket_urgency"].str.lower().map(urgency_map).fillna(1)
    tickets["sentiment_score"] = tickets["support_ticket_sentiment"].str.lower().map(sentiment_map).fillna(2)
    ticket_by_order = tickets.groupby("order_id", as_index=False).agg(
        max_ticket_urgency=("urgency_score", "max"),
        avg_ticket_sentiment=("sentiment_score", "mean"),
        ticket_count=("ticket_id", "count"),
    )

    orders_enriched = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(ticket_by_order, on="order_id", how="left")
        .merge(
            inventory[
                [
                    "sku",
                    "inventory_available_units",
                    "inventory_reserved_units",
                    "inventory_incoming_units",
                    "product_page_views_last_hour",
                    "conversion_rate_last_hour",
                ]
            ],
            on="sku",
            how="left",
        )
    )
    orders_enriched["ticket_count"] = orders_enriched["ticket_count"].fillna(0)
    orders_enriched["max_ticket_urgency"] = orders_enriched["max_ticket_urgency"].fillna(0)
    orders_enriched["is_vip"] = orders_enriched["is_vip"].fillna(False)
    orders_enriched["customer_lifetime_value"] = pd.to_numeric(
        orders_enriched["customer_lifetime_value"], errors="coerce"
    ).fillna(0)

    inv = inventory.copy()
    inv["stock_pressure"] = inv["inventory_reserved_units"] / (inv["inventory_available_units"] + 1)
    inv["demand_pressure"] = inv["product_page_views_last_hour"] * inv["conversion_rate_last_hour"]
    inv["stockout_risk"] = (
        0.55 * inv["stock_pressure"].clip(upper=10)
        + 0.45 * (inv["demand_pressure"] / (inv["inventory_available_units"] + 1)).clip(upper=10)
    )

    camp = campaigns.copy()
    intensity_map = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
    camp["intensity_score"] = camp["campaign_intensity"].str.lower().map(intensity_map).fillna(1)
    camp = camp.merge(
        inv[["sku", "stockout_risk", "inventory_available_units", "inventory_incoming_units"]],
        left_on="target_sku",
        right_on="sku",
        how="left",
    )
    camp["campaign_risk"] = (
        0.5 * camp["intensity_score"]
        + 0.35 * camp["stockout_risk"].fillna(0)
        + 0.15 * camp["traffic_growth"].fillna(0)
    )

    orders_enriched = add_ml_order_risk(orders_enriched)
    return {"orders_enriched": orders_enriched, "inventory_signals": inv, "campaign_signals": camp}


def add_ml_order_risk(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Train a lightweight ML model with weak supervision to estimate escalation risk.
    This is practical for hackathon settings where explicit labels are unavailable.
    """
    df = orders_enriched.copy()
    if "customer_segment" not in df.columns:
        if "customer_segment_x" in df.columns:
            df["customer_segment"] = df["customer_segment_x"]
        elif "customer_segment_y" in df.columns:
            df["customer_segment"] = df["customer_segment_y"]
        else:
            df["customer_segment"] = "unknown"
    df["is_vip_int"] = df["is_vip"].fillna(False).astype(int)
    df["is_express"] = df["shipping_method"].astype(str).str.lower().eq("express").astype(int)
    df["inventory_available_units"] = pd.to_numeric(df["inventory_available_units"], errors="coerce")
    df["inventory_reserved_units"] = pd.to_numeric(df["inventory_reserved_units"], errors="coerce")
    df["order_value_clean"] = pd.to_numeric(df["order_value_clean"], errors="coerce")
    df["customer_lifetime_value"] = pd.to_numeric(df["customer_lifetime_value"], errors="coerce")
    df["customer_orders_count"] = pd.to_numeric(df.get("customer_orders_count"), errors="coerce")
    df["customer_returns_count"] = pd.to_numeric(df.get("customer_returns_count"), errors="coerce")
    df["product_page_views_last_hour"] = pd.to_numeric(df["product_page_views_last_hour"], errors="coerce")
    df["conversion_rate_last_hour"] = pd.to_numeric(df["conversion_rate_last_hour"], errors="coerce")

    # Weak labels: escalation-like outcomes inferred from support and fragility signals.
    weak_label = (
        (df["max_ticket_urgency"] >= 3)
        | ((df["ticket_count"] > 0) & (df["avg_ticket_sentiment"] >= 2.6))
        | (
            df["order_status"].astype(str).str.lower().eq("payment_review")
            & df["shipping_method"].astype(str).str.lower().eq("express")
        )
    ).astype(int)

    numeric_features = [
        "order_value_clean",
        "inventory_available_units",
        "inventory_reserved_units",
        "customer_lifetime_value",
        "customer_orders_count",
        "customer_returns_count",
        "is_vip_int",
        "is_express",
        "product_page_views_last_hour",
        "conversion_rate_last_hour",
    ]
    categorical_features = ["order_status", "shipping_city", "shipping_method", "customer_segment", "campaign_source"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
        ]
    )

    # If weak labels collapse to a single class, fallback safely to heuristic estimate.
    if weak_label.nunique() < 2:
        heuristic = (
            0.35 * df["is_express"]
            + 0.30 * (df["inventory_reserved_units"] / (df["inventory_available_units"] + 1)).fillna(0)
            + 0.35 * (df["ticket_count"] > 0).astype(int)
        ).clip(0, 1)
        df["ml_escalation_risk"] = heuristic
        df["ml_risk_source"] = "fallback_heuristic"
        return df

    feature_cols = numeric_features + categorical_features
    model.fit(df[feature_cols], weak_label)
    proba = model.predict_proba(df[feature_cols])[:, 1]
    df["ml_escalation_risk"] = np.clip(proba, 0, 1)
    df["ml_risk_source"] = "logistic_weak_supervision"
    return df


def generate_actions(signals: Dict[str, Any]) -> List[Action]:
    actions: List[Action] = []
    orders = signals["orders_enriched"]
    inv = signals["inventory_signals"]
    camp = signals["campaign_signals"]

    # Campaign control actions
    risky_campaigns = camp.sort_values("campaign_risk", ascending=False).head(8)
    for _, row in risky_campaigns.iterrows():
        available = row.get("inventory_available_units", 0) or 0
        incoming = row.get("inventory_incoming_units", 0) or 0
        if row["campaign_risk"] > 4.3 and available <= 5:
            action_type = "pause_campaign"
            title = f"Pausar campaña {row['campaign_id']} para {row['target_sku']}"
            expected_impact = "Reducir oversell inmediato y presión de tickets en menos de 1 hora."
            revenue_penalty = 2.8
        elif row["campaign_risk"] > 3.5:
            action_type = "reduce_campaign_intensity"
            title = f"Reducir intensidad de {row['campaign_id']} y limitar ciudad objetivo"
            expected_impact = "Bajar presión de demanda manteniendo parte de conversión."
            revenue_penalty = 1.7
        else:
            action_type = "redirect_campaign"
            title = f"Redirigir {row['campaign_id']} a SKU alternativo con más stock"
            expected_impact = "Conservar tracción comercial sin comprometer cumplimiento."
            revenue_penalty = 1.0

        reason = (
            f"Campaña {row['campaign_source']} en {row['target_city']} sobre SKU {row['target_sku']} "
            f"con riesgo combinado {row['campaign_risk']:.2f} y stock disponible {int(available)}."
        )
        confidence = 0.55 + min(0.40, float(row["campaign_risk"]) / 10)
        actions.append(
            Action(
                action_type=action_type,
                target_id=str(row["campaign_id"]),
                title=title,
                reason=reason,
                expected_impact=expected_impact,
                confidence=round(min(confidence, 0.95), 2),
                owner="growth_ops",
                automation_possible=True,
                impact_score=3.4,
                effort_score=1.4 if action_type != "redirect_campaign" else 1.8,
                revenue_penalty=revenue_penalty,
                risk_reduction=min(5.0, float(row["campaign_risk"])),
                customer_protection=3.6,
                operational_cost=1.5,
            )
        )

    # Order-level manual prioritization actions
    orders_scored = orders.copy()
    orders_scored["logistics_risk"] = (
        (orders_scored["shipping_method"].str.lower().eq("express")).astype(float) * 1.8
        + (orders_scored["max_ticket_urgency"] * 0.9)
        + (orders_scored["ticket_count"] * 0.5)
        + (orders_scored["inventory_reserved_units"] / (orders_scored["inventory_available_units"] + 1))
    )
    orders_scored["customer_weight"] = (
        orders_scored["is_vip"].astype(float) * 2.3
        + (orders_scored["customer_lifetime_value"] / 400).clip(upper=3.0)
    )
    orders_scored["order_priority"] = (
        0.45 * orders_scored["logistics_risk"]
        + 0.30 * orders_scored["customer_weight"]
        + 0.25 * (orders_scored["ml_escalation_risk"].fillna(0) * 10)
    )

    critical_orders = orders_scored.sort_values("order_priority", ascending=False).head(14)
    for _, row in critical_orders.iterrows():
        is_vip = bool(row.get("is_vip", False))
        high_urgency = (row.get("max_ticket_urgency", 0) or 0) >= 3
        action_type = "prioritize_order" if is_vip or high_urgency else "manual_review_order"
        owner = "operations"
        title = f"Revisar de forma prioritaria pedido {row['order_id']}"
        reason = (
            f"Pedido {row['order_id']} con riesgo logístico {row['logistics_risk']:.2f}, "
            f"cliente {'VIP' if is_vip else row.get('customer_segment', 'sin segmento')}, "
            f"stock disponible del SKU {int((row.get('inventory_available_units', 0) or 0))}, "
            f"riesgo ML {float(row.get('ml_escalation_risk', 0)):.2f}."
        )
        actions.append(
            Action(
                action_type=action_type,
                target_id=str(row["order_id"]),
                title=title,
                reason=reason,
                expected_impact="Prevenir incidencias de fulfillment y reducir escalado de soporte.",
                confidence=round(min(0.92, 0.50 + float(row["order_priority"]) / 10), 2),
                owner=owner,
                automation_possible=False,
                impact_score=3.1,
                effort_score=2.3,
                revenue_penalty=0.4,
                risk_reduction=min(5.0, float(row["order_priority"])),
                customer_protection=4.1 if is_vip else 3.2,
                operational_cost=2.5,
            )
        )

    # Proactive customer communication actions
    comm_candidates = orders_scored[
        (orders_scored["max_ticket_urgency"] >= 3)
        | (orders_scored["shipping_method"].str.lower().eq("express") & (orders_scored["ticket_count"] > 0))
    ].sort_values(["max_ticket_urgency", "ticket_count"], ascending=False)
    for _, row in comm_candidates.head(6).iterrows():
        actions.append(
            Action(
                action_type="contact_customer_proactively",
                target_id=str(row["customer_id"]),
                title=f"Contactar proactivamente cliente {row['customer_id']}",
                reason=(
                    f"Cliente con pedido {row['order_id']} y señales de fricción (urgencia ticket "
                    f"{int(row.get('max_ticket_urgency', 0))}, envío {row.get('shipping_method', 'n/a')})."
                ),
                expected_impact="Reducir incertidumbre del cliente y prevenir reclamaciones posteriores.",
                confidence=0.78,
                owner="support",
                automation_possible=True,
                impact_score=2.7,
                effort_score=1.2,
                revenue_penalty=0.1,
                risk_reduction=2.9,
                customer_protection=4.4,
                operational_cost=1.0,
            )
        )

    # Inventory expedites for extreme stock pressure
    critical_skus = inv.sort_values("stockout_risk", ascending=False).head(5)
    for _, row in critical_skus.iterrows():
        if (row["inventory_available_units"] <= 6) and (row["inventory_incoming_units"] > 0):
            actions.append(
                Action(
                    action_type="expedite_incoming_stock",
                    target_id=str(row["sku"]),
                    title=f"Acelerar reposición entrante del SKU {row['sku']}",
                    reason=(
                        f"SKU con riesgo de rotura {row['stockout_risk']:.2f}, "
                        f"stock disponible {int(row['inventory_available_units'])} y demanda elevada."
                    ),
                    expected_impact="Reducir probabilidad de stockout en próximas horas de pico.",
                    confidence=0.81,
                    owner="supply_chain",
                    automation_possible=False,
                    impact_score=3.3,
                    effort_score=2.1,
                    revenue_penalty=0.2,
                    risk_reduction=min(5.0, float(row["stockout_risk"])),
                    customer_protection=3.8,
                    operational_cost=2.0,
                )
            )

    return actions


def select_portfolio(actions: List[Action], max_actions: int = 10) -> List[Action]:
    selected: List[Action] = []
    used_types: Dict[str, int] = {}

    for action in sorted(actions, key=lambda a: a.total_score, reverse=True):
        if len(selected) >= max_actions:
            break

        # Portfolio constraints to avoid one-dimensional recommendations.
        if used_types.get(action.action_type, 0) >= 3:
            continue
        if action.action_type == "pause_campaign" and used_types.get("pause_campaign", 0) >= 2:
            continue

        selected.append(action)
        used_types[action.action_type] = used_types.get(action.action_type, 0) + 1

    return selected


def to_output(actions: List[Action]) -> List[Dict[str, Any]]:
    output = []
    for idx, a in enumerate(actions, start=1):
        output.append(
            {
                "rank": idx,
                "action_type": a.action_type,
                "target_id": a.target_id,
                "title": a.title,
                "reason": a.reason,
                "expected_impact": a.expected_impact,
                "confidence": round(a.confidence, 2),
                "owner": a.owner,
                "automation_possible": a.automation_possible,
            }
        )
    return output


def main() -> None:
    data = load_data(DATA_DIR)
    signals = build_signals(data)
    candidates = generate_actions(signals)
    portfolio = select_portfolio(candidates, max_actions=10)
    payload = to_output(portfolio)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

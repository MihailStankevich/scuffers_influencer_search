from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from influencer_ops import build_influencer_recommendations
    from scuffers_engine import run_engine
except ModuleNotFoundError:
    from hackathon.influencer_ops import build_influencer_recommendations
    from hackathon.scuffers_engine import run_engine


app = FastAPI(title="Scuffers Ops API", version="1.0.0")
DATA_DIR = Path(__file__).parent


class LaunchRiskRequest(BaseModel):
    campaign_intensity: str = Field(default="medium", description="low|medium|high|very_high")
    expected_traffic_growth: float = Field(default=2.0, ge=0)
    expected_conversion_rate: float = Field(default=0.05, ge=0, le=1)
    available_units: int = Field(default=20, ge=0)
    reserved_units: int = Field(default=5, ge=0)
    incoming_units: int = Field(default=0, ge=0)
    incoming_eta_hours: float = Field(default=48, ge=0)
    vip_share: float = Field(default=0.1, ge=0, le=1)
    express_share: float = Field(default=0.2, ge=0, le=1)
    current_support_load: int = Field(default=5, ge=0)


def _compute_launch_risk(payload: LaunchRiskRequest) -> Dict[str, float]:
    intensity_map = {"low": 1.0, "medium": 2.0, "high": 3.0, "very_high": 4.0}
    intensity_score = intensity_map.get(payload.campaign_intensity.lower(), 2.0)
    stock_stress = payload.reserved_units / (payload.available_units + 1)
    demand_pressure = (payload.expected_traffic_growth * payload.expected_conversion_rate * 100) / (
        payload.available_units + 1
    )
    incoming_relief = payload.incoming_units / (payload.incoming_eta_hours + 1)
    customer_complexity = payload.vip_share * 2.0 + payload.express_share * 1.6
    support_pressure = payload.current_support_load / 10

    raw = (
        0.22 * intensity_score
        + 0.24 * min(stock_stress, 10)
        + 0.20 * min(demand_pressure, 10)
        + 0.14 * min(customer_complexity, 10)
        + 0.16 * min(support_pressure, 10)
        - 0.12 * min(incoming_relief, 10)
    )
    risk_0_100 = max(0.0, min(100.0, raw * 11))
    return {
        "risk_score": round(risk_0_100, 2),
        "stock_stress": round(stock_stress, 3),
        "demand_pressure": round(demand_pressure, 3),
        "incoming_relief": round(incoming_relief, 3),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status(candidate_id: Optional[str] = None) -> Dict[str, object]:
    result = run_engine(DATA_DIR, candidate_id=candidate_id, shipping_limit=12)
    actions = pd.DataFrame(result["actions"])
    city_issues = result["city_issues"]
    sku_table = result["sku_table"]
    shipping_statuses = result["shipping_statuses"]
    hottest_city = (
        city_issues.groupby("shipping_city", as_index=False)["operational_heat"].max().sort_values("operational_heat", ascending=False).head(1)
    )
    top_sku = sku_table.sort_values("sku_risk", ascending=False).head(1)
    return {
        "top_actions": actions.head(5).to_dict(orient="records"),
        "hottest_city": hottest_city.to_dict(orient="records"),
        "critical_sku": top_sku[["sku", "sku_risk", "inventory_available_units"]].to_dict(orient="records"),
        "shipping_api": {
            "queried_orders": int(len(shipping_statuses)),
            "max_shipping_api_risk": float(shipping_statuses["shipping_api_risk"].max()) if not shipping_statuses.empty else 0,
        },
    }


@app.get("/dashboard")
def dashboard(issue_type: Optional[str] = None, customer_segment: Optional[str] = None) -> Dict[str, object]:
    result = run_engine(DATA_DIR)
    city_issues = result["city_issues"]
    service_routes = result["service_routes"]

    if issue_type:
        city_issues = city_issues[city_issues["issue_type"] == issue_type]
    if customer_segment:
        service_routes = service_routes[service_routes["customer_segment"] == customer_segment]
        allowed_cities = set(service_routes["shipping_city"].dropna().tolist())
        city_issues = city_issues[city_issues["shipping_city"].isin(allowed_cities)]

    city_summary = (
        city_issues.groupby("shipping_city", as_index=False)
        .agg(total_tickets=("tickets", "sum"), operational_heat=("operational_heat", "max"))
        .sort_values("operational_heat", ascending=False)
    )
    return {
        "filters": {"issue_type": issue_type, "customer_segment": customer_segment},
        "city_summary": city_summary.to_dict(orient="records"),
        "routes_summary": service_routes.groupby("service_route", as_index=False)["ticket_id"].count().to_dict(orient="records"),
    }


@app.post("/risk/launch")
def launch_risk(payload: LaunchRiskRequest) -> Dict[str, object]:
    risk = _compute_launch_risk(payload)
    score = risk["risk_score"]
    if score >= 70:
        recommendation = "no_go"
        action = "Pausar o reducir intensidad y reforzar soporte antes de lanzar."
    elif score >= 45:
        recommendation = "caution"
        action = "Lanzar con límites geográficos y monitorización cada 15 minutos."
    else:
        recommendation = "go"
        action = "Lanzamiento viable con monitorización estándar."

    return {
        "risk": risk,
        "recommendation": recommendation,
        "suggested_action": action,
    }


@app.get("/actions")
def actions(limit: int = 10, candidate_id: Optional[str] = None) -> Dict[str, List[Dict[str, object]]]:
    result = run_engine(DATA_DIR, candidate_id=candidate_id, shipping_limit=12)
    actions_df = pd.DataFrame(result["actions"]).head(max(1, min(limit, 20)))
    return {"actions": actions_df.to_dict(orient="records")}


@app.get("/creators/recommendations")
def creator_recommendations(limit: int = 10) -> Dict[str, List[Dict[str, object]]]:
    recs = build_influencer_recommendations(DATA_DIR, max_rows=max(1, min(limit, 20)))
    return {"recommendations": recs.to_dict(orient="records")}

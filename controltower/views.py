from __future__ import annotations

import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from .services import DEFAULT_CANDIDATE_ID, get_ops_context, run_creator_match


def _compute_launch_risk(payload: dict) -> dict:
    intensity_map = {"low": 1.0, "medium": 2.0, "high": 3.0, "very_high": 4.0}
    intensity = intensity_map.get(str(payload.get("campaign_intensity", "medium")).lower(), 2.0)
    available = max(0.0, float(payload.get("available_units", 20) or 0))
    reserved = max(0.0, float(payload.get("reserved_units", 5) or 0))
    incoming = max(0.0, float(payload.get("incoming_units", 0) or 0))
    eta = max(0.0, float(payload.get("incoming_eta_hours", 48) or 0))
    traffic = max(0.0, float(payload.get("expected_traffic_growth", 2.0) or 0))
    conversion = min(1.0, max(0.0, float(payload.get("expected_conversion_rate", 0.05) or 0)))
    vip_share = min(1.0, max(0.0, float(payload.get("vip_share", 0.1) or 0)))
    express_share = min(1.0, max(0.0, float(payload.get("express_share", 0.2) or 0)))
    support_load = max(0.0, float(payload.get("current_support_load", 5) or 0))

    stock_stress = reserved / (available + 1)
    demand_pressure = (traffic * conversion * 100) / (available + 1)
    incoming_relief = incoming / (eta + 1)
    customer_complexity = vip_share * 2.0 + express_share * 1.6
    support_pressure = support_load / 10
    raw = (
        0.22 * intensity
        + 0.24 * min(stock_stress, 10)
        + 0.20 * min(demand_pressure, 10)
        + 0.14 * min(customer_complexity, 10)
        + 0.16 * min(support_pressure, 10)
        - 0.12 * min(incoming_relief, 10)
    )
    score = max(0.0, min(100.0, raw * 11))
    if score >= 70:
        recommendation = "no_go"
        suggested_action = "Pausar o reducir intensidad y reforzar soporte antes de lanzar."
    elif score >= 45:
        recommendation = "caution"
        suggested_action = "Lanzar con límites geográficos y monitorización cada 15 minutos."
    else:
        recommendation = "go"
        suggested_action = "Lanzamiento viable con monitorización estándar."
    return {
        "risk_score": round(score, 2),
        "stock_stress": round(stock_stress, 3),
        "demand_pressure": round(demand_pressure, 3),
        "incoming_relief": round(incoming_relief, 3),
        "recommendation": recommendation,
        "suggested_action": suggested_action,
    }


def dashboard(request):
    context = get_ops_context(candidate_id=DEFAULT_CANDIDATE_ID, fetch_shipping=False)
    risk_input = {
        "campaign_intensity": request.POST.get("campaign_intensity", "high"),
        "expected_traffic_growth": request.POST.get("expected_traffic_growth", "2.8"),
        "expected_conversion_rate": request.POST.get("expected_conversion_rate", "0.06"),
        "available_units": request.POST.get("available_units", "8"),
        "reserved_units": request.POST.get("reserved_units", "18"),
        "incoming_units": request.POST.get("incoming_units", "0"),
        "incoming_eta_hours": request.POST.get("incoming_eta_hours", "48"),
        "vip_share": request.POST.get("vip_share", "0.18"),
        "express_share": request.POST.get("express_share", "0.35"),
        "current_support_load": request.POST.get("current_support_load", "12"),
    }
    context["risk_input"] = risk_input
    context["risk_result"] = _compute_launch_risk(risk_input)
    return render(request, "controltower/dashboard.html", context)


def creator_match(request):
    result = None
    error = None
    if request.method == "POST":
        uploaded_file = request.FILES.get("image")
        if not uploaded_file:
            error = "Sube una imagen de producto para buscar influencers."
        else:
            try:
                result = run_creator_match(
                    uploaded_file=uploaded_file,
                    mode=request.POST.get("mode", "business"),
                    city=request.POST.get("city", ""),
                    country=request.POST.get("country", ""),
                    keywords=request.POST.get("keywords", "streetwear,oversized,hoodie"),
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

    return render(
        request,
        "controltower/match.html",
        {
            "result": result,
            "error": error,
            "mode": request.POST.get("mode", "business"),
            "city": request.POST.get("city", ""),
            "country": request.POST.get("country", ""),
            "keywords": request.POST.get("keywords", "streetwear,oversized,hoodie"),
        },
    )


@require_GET
def actions_api(request):
    candidate_id = request.GET.get("candidate_id", DEFAULT_CANDIDATE_ID)
    context = get_ops_context(candidate_id=candidate_id, fetch_shipping=False)
    return JsonResponse({"actions": context["actions"]})


@require_http_methods(["POST"])
def launch_risk_api(request):
    payload = {}
    if request.content_type and "application/json" in request.content_type.lower():
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = request.POST.dict()

    result = _compute_launch_risk(payload)
    return JsonResponse(
        {
            "risk": {
                "risk_score": result["risk_score"],
                "stock_stress": result["stock_stress"],
                "demand_pressure": result["demand_pressure"],
                "incoming_relief": result["incoming_relief"],
            },
            "recommendation": result["recommendation"],
            "suggested_action": result["suggested_action"],
        }
    )


@require_GET
def service_worker(request):
    return HttpResponse("// no service worker for local Django demo\n", content_type="application/javascript")

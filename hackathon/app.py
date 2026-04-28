from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from scuffers_engine import run_engine
except ModuleNotFoundError:
    from hackathon.scuffers_engine import run_engine

try:
    import pydeck as pdk

    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

try:
    import plotly.express as px

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


st.set_page_config(page_title="Scuffers Control Tower", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 18% 8%, rgba(26, 95, 65, 0.18), transparent 28%),
            linear-gradient(180deg, #070909 0%, #050606 100%);
        color: #f5f7f6;
    }
    .block-container {padding-top: 2rem; padding-bottom: 2.2rem; max-width: 1040px;}
    h1 {font-size: 2.35rem; font-weight: 750; letter-spacing: -0.045em; margin: 0;}
    h2, h3 {letter-spacing: -0.025em;}
    p, span, label {color: #eef3f1;}
    .scuffers-header {
        display: flex;
        align-items: center;
        gap: 22px;
        margin-bottom: 26px;
    }
    .scuffers-logo {
        width: 68px;
        height: 68px;
        border-radius: 18px;
        object-fit: cover;
        box-shadow: 0 18px 55px rgba(0,0,0,0.35);
    }
    .scuffers-kicker {
        color: rgba(245,247,246,0.56);
        font-size: 0.86rem;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);}
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 10px 4px;
        color: rgba(245,247,246,0.68);
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom: 2px solid #207c54;
    }
    .stMetric {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        backdrop-filter: blur(14px);
    }
    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        overflow: hidden;
    }
    [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
    }
    [data-testid="stButton"] button {
        background: #207c54;
        color: #ffffff;
        border: 0;
        border-radius: 12px;
        padding: 0.55rem 0.9rem;
    }
    [data-testid="stButton"] button:hover {background: #2a9868;}
    section[data-testid="stSidebar"] {background: #050606;}
    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.085);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(16px);
    }
    .small-label {
        color: rgba(245,247,246,0.58);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .big-value {font-size: 1.55rem; font-weight: 720; letter-spacing: -0.03em;}
    .muted {color: rgba(245,247,246,0.62); font-size: 0.9rem;}
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(32,124,84,0.18);
        border: 1px solid rgba(54,178,122,0.22);
        color: #c8f5df;
        font-size: 0.78rem;
        margin-right: 6px;
    }
    .danger-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255,60,60,0.14);
        border: 1px solid rgba(255,80,80,0.24);
        color: #ffd0d0;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_path = Path(__file__).parent / "channels4_profile.jpg"
logo_html = f'<img class="scuffers-logo" src="{logo_path.as_uri()}">' if logo_path.exists() else ""
st.markdown(
    f"""
    <div class="scuffers-header">
        {logo_html}
        <div>
            <h1>Scuffers Control Tower</h1>
            <div class="scuffers-kicker">Launch operations, support pressure and regional heat.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

data_dir = Path(__file__).parent
candidate_id = "#SCF-2026-6495"
fetch_shipping = True
shipping_limit = 18
creator_match_url = "http://127.0.0.1:8000/ui?v=mini"

result = run_engine(
    data_dir,
    candidate_id=candidate_id or None,
    fetch_shipping=fetch_shipping,
    shipping_limit=shipping_limit,
)

actions_df = pd.DataFrame(result["actions"])
sku_df = result["sku_table"].sort_values("sku_risk", ascending=False).reset_index(drop=True)
corr_df = result["correlations"]
city_issues_df = result["city_issues"]
service_routes_df = result["service_routes"]
shipping_status_df = result["shipping_statuses"]
orders_raw_df = pd.read_csv(data_dir / "orders.csv")
inventory_raw_df = pd.read_csv(data_dir / "inventory.csv")
orders_raw_df["quantity"] = pd.to_numeric(orders_raw_df["quantity"], errors="coerce").fillna(0)
product_demand_df = (
    inventory_raw_df[["sku", "product_name"]]
    .merge(
        orders_raw_df.groupby("sku", as_index=False).agg(
            quantity_ordered=("quantity", "sum"),
            orders=("order_id", "count"),
        ),
        on="sku",
        how="left",
    )
    .fillna({"quantity_ordered": 0, "orders": 0})
)
product_demand_df["quantity_ordered"] = product_demand_df["quantity_ordered"].astype(int)
product_demand_df["orders"] = product_demand_df["orders"].astype(int)
product_demand_df = product_demand_df.sort_values(["quantity_ordered", "orders", "product_name"], ascending=True).head(10)

issue_options = ["Todos"] + sorted(city_issues_df["issue_type"].dropna().unique().tolist())
customer_type_options = ["Todos"] + sorted(service_routes_df["customer_segment"].dropna().unique().tolist())
issue_labels = {
    "cambio_direccion": "Cambio dirección",
    "confirmacion_entrega": "Confirmación entrega",
    "confirmacion_plazo": "Confirmación plazo",
    "miedo_rotura_stock": "Miedo stockout",
    "no_movimiento_envio": "Sin movimiento envío",
    "solicitud_prioridad": "Solicitud prioridad",
    "otros": "Otros",
}
route_labels = {
    "chatbot_first": "Chatbot / playbook",
    "white_glove_human": "Agente senior",
    "human_priority": "Humano prioritario",
}

filters_col1, filters_col2 = st.columns(2)
with filters_col1:
    selected_issue = st.selectbox("Tipo de incidencia", issue_options)
with filters_col2:
    selected_customer_type = st.selectbox("Tipo de cliente", customer_type_options)

filtered_city_issues = city_issues_df.copy()
if selected_issue != "Todos":
    filtered_city_issues = filtered_city_issues[filtered_city_issues["issue_type"] == selected_issue]

filtered_service_routes = service_routes_df.copy()
if selected_customer_type != "Todos":
    filtered_service_routes = filtered_service_routes[filtered_service_routes["customer_segment"] == selected_customer_type]
allowed_cities = set(filtered_service_routes["shipping_city"].dropna().unique().tolist())
if allowed_cities:
    filtered_city_issues = filtered_city_issues[filtered_city_issues["shipping_city"].isin(allowed_cities)]

city_coords = {
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

city_map_df = (
    filtered_city_issues.groupby("shipping_city", as_index=False)
    .agg(
        total_tickets=("tickets", "sum"),
        operational_heat=("operational_heat", "max"),
        avg_city_urgency=("avg_city_urgency", "max"),
    )
    .sort_values("operational_heat", ascending=False)
)
city_map_df["lat"] = city_map_df["shipping_city"].map(lambda c: city_coords.get(c, (None, None))[0])
city_map_df["lon"] = city_map_df["shipping_city"].map(lambda c: city_coords.get(c, (None, None))[1])
city_map_df = city_map_df.dropna(subset=["lat", "lon"])
if not city_map_df.empty:
    max_heat = city_map_df["operational_heat"].max()
    city_map_df["heat_norm"] = city_map_df["operational_heat"] / max_heat if max_heat else 0
    city_map_df["radius"] = 18000 + city_map_df["total_tickets"] * 8500
    city_map_df["color"] = city_map_df["heat_norm"].apply(
        lambda v: [255, int(80 * (1 - v)), int(70 * (1 - v)), 190]
    )

tab_dashboard, tab_creators, tab_customers, tab_regions, tab_shipping, tab_api_risk, tab_stock = st.tabs(
    ["Dashboard", "Creators Growth", "Clientes", "Regiones", "Shipping API", "API Risk", "Stock & Signals"]
)

with tab_dashboard:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidencias filtradas", int(filtered_city_issues["tickets"].sum()) if not filtered_city_issues.empty else 0)
    m2.metric("Ciudades críticas", int(city_map_df.shape[0]))
    m3.metric("Clientes en scope", int(filtered_service_routes["customer_id"].nunique()))
    m4.metric("Acciones activas", len(actions_df))

    map_col, insight_col = st.columns([1.55, 1])
    with map_col:
        st.subheader("Mapa operativo general")
        st.caption("Hover sobre cada punto para ver ciudad, incidencias y heat operacional.")
        if HAS_PYDECK and not city_map_df.empty:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=city_map_df,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
            view_state = pdk.ViewState(latitude=40.2, longitude=-3.7, zoom=5.05, pitch=0)
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={
                    "html": (
                        "<b>{shipping_city}</b><br/>"
                        "Incidencias: {total_tickets}<br/>"
                        "Heat: {operational_heat}<br/>"
                        "Urgencia media: {avg_city_urgency}"
                    ),
                    "style": {
                        "backgroundColor": "rgba(5, 8, 8, 0.94)",
                        "color": "white",
                        "border": "1px solid rgba(32,124,84,0.8)",
                        "borderRadius": "10px",
                    },
                },
            )
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.map(city_map_df.rename(columns={"lat": "latitude", "lon": "longitude"}))

    with insight_col:
        st.subheader("Dónde actuar primero")
        if not city_map_df.empty:
            for idx, city in city_map_df.head(3).reset_index(drop=True).iterrows():
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="small-label">#{idx + 1} ciudad crítica</div>
                        <div class="big-value">{city['shipping_city']}</div>
                        <div class="muted">{int(city['total_tickets'])} incidencias · heat {float(city['operational_heat']):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        top_issues = (
            filtered_city_issues.groupby("issue_type", as_index=False)["tickets"]
            .sum()
            .sort_values("tickets", ascending=False)
            .head(3)
        )
        st.markdown('<div class="small-label">Incidencias dominantes</div>', unsafe_allow_html=True)
        for _, issue in top_issues.iterrows():
            st.markdown(
                f'<span class="pill">{issue_labels.get(issue["issue_type"], issue["issue_type"])} · {int(issue["tickets"])}</span>',
                unsafe_allow_html=True,
            )

        if not actions_df.empty:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="small-label">Siguiente acción</div>
                    <div style="font-weight:680;">{actions_df.iloc[0]['title']}</div>
                    <div class="muted">{actions_df.iloc[0]['owner']} · importancia {actions_df.iloc[0]['importance_score']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Generador de reportes")
    top_city = city_map_df.iloc[0]["shipping_city"] if not city_map_df.empty else "N/A"
    top_heat = float(city_map_df.iloc[0]["operational_heat"]) if not city_map_df.empty else 0.0
    report_md = (
        "# Scuffers Ops Report\n\n"
        f"- Incidencias filtradas: {int(filtered_city_issues['tickets'].sum()) if not filtered_city_issues.empty else 0}\n"
        f"- Ciudad más crítica: {top_city} (heat {top_heat:.2f})\n"
        f"- Clientes en alcance: {int(filtered_service_routes['customer_id'].nunique())}\n"
        f"- Top acción: {actions_df.iloc[0]['title'] if not actions_df.empty else 'N/A'}\n\n"
        "## Top 5 acciones\n"
        + "\n".join([f"- {row['title']} ({row['owner']})" for _, row in actions_df.head(5).iterrows()])
    )
    st.download_button(
        "Descargar reporte (Markdown)",
        data=report_md,
        file_name="scuffers_ops_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "Descargar acciones en JSON",
        data=actions_df.to_json(orient="records", force_ascii=False, indent=2),
        file_name="scuffers_top10_actions.json",
        mime="application/json",
    )

with tab_customers:
    st.subheader("Gestión de clientes (prioridad de atención)")
    route_summary = (
        filtered_service_routes.groupby("service_route", as_index=False)
        .agg(tickets=("ticket_id", "count"), avg_priority=("service_priority", "mean"), avg_clv=("customer_lifetime_value", "mean"))
        .sort_values("tickets", ascending=False)
    )
    route_summary["service_route"] = route_summary["service_route"].replace(route_labels)
    route_summary = route_summary.rename(
        columns={"service_route": "Ruta", "tickets": "Tickets", "avg_priority": "Prioridad media", "avg_clv": "CLV medio"}
    )
    st.dataframe(route_summary, use_container_width=True)

    clients_view = (
        filtered_service_routes.sort_values("service_priority", ascending=False)
        .drop_duplicates(subset=["customer_id"])
        .head(16)
    )
    for _, row in clients_view.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.55, 1.05, 1.0, 1.1])
            route = route_labels.get(row["service_route"], row["service_route"])
            value_tier = "Alto valor" if bool(row["is_vip"]) or float(row["customer_lifetime_value"]) >= 600 or int(row["customer_orders_count"]) >= 8 else "Bajo histórico"
            operational_tier = "Alta urgencia" if float(row["sku_risk"]) >= 7.5 or int(row["urgency_score"]) >= 3 else "Normal"
            c1.markdown(
                f"**{row['customer_id']}** · {row['customer_segment']}  \n"
                f"CLV: `{float(row['customer_lifetime_value']):.0f}` · VIP: `{bool(row['is_vip'])}`"
            )
            c2.write(f"Ruta: `{route}`")
            c3.write(f"{value_tier}  \n{operational_tier}  \nPrioridad `{float(row['service_priority']):.2f}`")
            support_channel = str(row.get("channel", "support")).strip() or "support"
            with c4:
                if st.button(f"Conectar por {support_channel}", key=f"connect_{row['ticket_id']}"):
                    st.success(f"Workflow de contacto abierto para {row['customer_id']} por canal `{support_channel}`.")

with tab_regions:
    st.subheader("Tipos de incidencia por ciudad")
    issues_pivot = (
        filtered_city_issues.pivot_table(
            index="shipping_city",
            columns="issue_type",
            values="tickets",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    if HAS_PLOTLY and not issues_pivot.empty:
        fig_bar = px.bar(
            issues_pivot,
            x="shipping_city",
            y=[c for c in issues_pivot.columns if c != "shipping_city"],
            color_discrete_sequence=px.colors.sequential.Reds_r,
        )
        fig_bar.update_layout(barmode="stack", margin=dict(l=8, r=8, t=8, b=8), legend_title_text="issue_type")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.bar_chart(issues_pivot.set_index("shipping_city") if not issues_pivot.empty else pd.DataFrame())

    st.dataframe(
        city_map_df.style.background_gradient(subset=["operational_heat"], cmap="Reds"),
        use_container_width=True,
    )

with tab_shipping:
    st.subheader("Shipping Status API: priorización en caliente")
    st.caption(
        "El motor consulta la API solo para pedidos que ya salían relevantes por stock, VIP, express o ticket. "
        "Si la API devuelve retraso, excepción o revisión manual, el pedido sube en la lista de acciones."
    )
    if not candidate_id:
        st.warning("Añade tu `X-Candidate-Id` en la barra lateral para activar la consulta real.")
    elif shipping_status_df.empty:
        st.info("No hay respuestas de Shipping API para mostrar todavía.")
    else:
        ok_count = int((shipping_status_df["shipping_api_status"] == "ok").sum())
        err_count = int((shipping_status_df["shipping_api_status"] != "ok").sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("Pedidos consultados", len(shipping_status_df))
        s2.metric("Respuestas OK", ok_count)
        s3.metric("Errores controlados", err_count)

        display_cols = [
            "order_id",
            "shipping_status",
            "delay_risk",
            "delay_reason",
            "requires_manual_review",
            "shipping_api_risk",
            "pre_api_order_pressure",
            "shipping_api_status",
        ]
        st.dataframe(
            shipping_status_df[display_cols].style.background_gradient(subset=["shipping_api_risk"], cmap="Reds"),
            use_container_width=True,
        )
        top_shipping = shipping_status_df[shipping_status_df["shipping_api_risk"] > 0].head(3)
        st.markdown("**Cómo cambió la decisión**")
        if top_shipping.empty:
            st.write("La API no encontró retrasos fuertes entre los pedidos consultados; se mantiene la priorización inicial.")
        for _, row in top_shipping.iterrows():
            st.markdown(
                f"- `{row['order_id']}` sube por estado `{row['shipping_status']}`, "
                f"`delay_risk={float(row['delay_risk']):.2f}` y motivo `{row.get('delay_reason') or 'n/a'}`."
            )

with tab_creators:
    st.subheader("Creators Growth")
    st.caption(
        "Scuffers tiene una estetica muy marcada. Colaborar con creadores visualmente alineados "
        "suele ser mas eficiente que invertir solo en perfiles masivos. "
        "En el matching puedes elegir modo Style only o Business y filtrar por pais para crecer por region."
    )
    creator_match_url = creator_match_url.strip() or "http://127.0.0.1:8000/ui?v=mini"

    header = st.columns([2.4, 1, 1.25])
    header[0].markdown("**Producto**")
    header[1].markdown("**Cantidad**")
    header[2].markdown("**Acción**")
    for _, row in product_demand_df.iterrows():
        c1, c2, c3 = st.columns([2.4, 1, 1.25])
        c1.write(str(row["product_name"]))
        c2.write(int(row["quantity_ordered"]))
        c3.link_button("Ver influencers", creator_match_url, use_container_width=True)

    st.caption("La búsqueda se hace en la otra página para mantener aquí solo la señal operativa.")

with tab_api_risk:
    st.subheader("Probar API de riesgo de lanzamiento")
    st.caption("Simula el endpoint `POST /risk/launch`: introduces parámetros y devuelve riesgo, recomendación y acción.")

    c1, c2, c3 = st.columns(3)
    with c1:
        campaign_intensity = st.selectbox("Intensidad campaña", ["low", "medium", "high", "very_high"], index=2)
        expected_traffic_growth = st.number_input("Traffic growth esperado", min_value=0.0, value=2.8, step=0.1)
        expected_conversion_rate = st.number_input("Conversion rate esperado", min_value=0.0, max_value=1.0, value=0.06, step=0.01)
    with c2:
        available_units = st.number_input("Stock disponible", min_value=0, value=8, step=1)
        reserved_units = st.number_input("Unidades reservadas", min_value=0, value=18, step=1)
        incoming_units = st.number_input("Stock entrante", min_value=0, value=0, step=1)
    with c3:
        incoming_eta_hours = st.number_input("ETA stock entrante (horas)", min_value=0.0, value=48.0, step=1.0)
        vip_share = st.slider("% VIP / alto valor", 0.0, 1.0, 0.18, 0.01)
        express_share = st.slider("% pedidos express", 0.0, 1.0, 0.35, 0.01)
        current_support_load = st.number_input("Tickets abiertos actuales", min_value=0, value=12, step=1)

    intensity_map = {"low": 1.0, "medium": 2.0, "high": 3.0, "very_high": 4.0}
    intensity_score = intensity_map[campaign_intensity]
    stock_stress = reserved_units / (available_units + 1)
    demand_pressure = (expected_traffic_growth * expected_conversion_rate * 100) / (available_units + 1)
    incoming_relief = incoming_units / (incoming_eta_hours + 1)
    customer_complexity = vip_share * 2.0 + express_share * 1.6
    support_pressure = current_support_load / 10
    raw_risk = (
        0.22 * intensity_score
        + 0.24 * min(stock_stress, 10)
        + 0.20 * min(demand_pressure, 10)
        + 0.14 * min(customer_complexity, 10)
        + 0.16 * min(support_pressure, 10)
        - 0.12 * min(incoming_relief, 10)
    )
    risk_score = round(max(0.0, min(100.0, raw_risk * 11)), 2)
    if risk_score >= 70:
        recommendation = "no_go"
        suggested_action = "Pausar o reducir intensidad y reforzar soporte antes de lanzar."
        risk_label = "NO GO"
        risk_color = "#ff4d4d"
    elif risk_score >= 45:
        recommendation = "caution"
        suggested_action = "Lanzar con límites geográficos y monitorización cada 15 minutos."
        risk_label = "CAUTION"
        risk_color = "#ffb020"
    else:
        recommendation = "go"
        suggested_action = "Lanzamiento viable con monitorización estándar."
        risk_label = "GO"
        risk_color = "#2a9868"

    response = {
        "risk": {
            "risk_score": risk_score,
            "stock_stress": round(stock_stress, 3),
            "demand_pressure": round(demand_pressure, 3),
            "incoming_relief": round(incoming_relief, 3),
        },
        "recommendation": recommendation,
        "suggested_action": suggested_action,
    }
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="small-label">Resultado simulación</div>
            <div style="display:flex;align-items:end;gap:18px;">
                <div class="big-value">{risk_score}/100</div>
                <div class="danger-pill" style="border-color:{risk_color}; color:{risk_color};">{risk_label}</div>
            </div>
            <div class="muted">{suggested_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(risk_score))
    st.json(response)
    st.code(
        """curl -X POST http://localhost:8000/risk/launch \\
  -H "Content-Type: application/json" \\
  -d '{ "campaign_intensity": "high", "expected_traffic_growth": 2.8, "available_units": 8 }'""",
        language="bash",
    )

with tab_stock:
    st.subheader("Radar de SKU risk")
    st.dataframe(
        sku_df[
            [
                "sku",
                "product_name",
                "inventory_available_units",
                "inventory_reserved_units",
                "ticket_count",
                "stock_stress",
                "demand_stress",
                "incoming_relief",
                "sku_risk",
            ]
        ].style.background_gradient(subset=["sku_risk"], cmap="Reds"),
        use_container_width=True,
    )
    st.subheader("Correlaciones automáticas con SKU risk")
    st.dataframe(corr_df, use_container_width=True)

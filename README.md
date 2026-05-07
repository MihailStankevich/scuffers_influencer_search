# Scuffers Control Tower (Django)

Proyecto principal en Django para operar un lanzamiento de alta demanda:

- Dashboard operativo con resumen, mapa de incidencias, clientes prioritarios, productos y acciones top 10.
- Style Matching de influencers en `/match/` (modo `Style` y `Business side`).
- API interna para riesgo de campaña (`POST /api/risk/launch`).

## 1) Requisitos

- Python 3.10+ (recomendado 3.10/3.11)
- Windows / macOS / Linux
- Internet en la primera ejecución (OpenCLIP puede descargar pesos)

## 2) Instalación

Desde la raíz del repo:

```bash
python -m venv .venv
```

Activar entorno:

- Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux
```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Aplicar migraciones Django:

```bash
python manage.py migrate
```

## 3) Ejecutar app (un solo servidor)

```bash
python manage.py runserver
```

Abrir:

- Home / Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Style Matching: [http://127.0.0.1:8000/match/](http://127.0.0.1:8000/match/)

## 4) Endpoints API

- `GET /api/actions/` -> acciones priorizadas (top 10)
- `POST /api/risk/launch` -> cálculo de riesgo de campaña

Ejemplo `POST /api/risk/launch`:

```bash
curl -X POST http://127.0.0.1:8000/api/risk/launch \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_intensity":"high",
    "expected_traffic_growth":2.8,
    "expected_conversion_rate":0.06,
    "available_units":8,
    "reserved_units":18,
    "incoming_units":0,
    "incoming_eta_hours":48,
    "vip_share":0.18,
    "express_share":0.35,
    "current_support_load":12
  }'
```

## 5) Flujo demo recomendado

1. Abrir `/` y revisar el resumen operativo.
2. Usar pestañas para profundizar en clientes, regiones, productos y stock.
3. En `API Risk`, probar escenarios de campaña.
4. En productos/clientes, ir a `/match/` y lanzar búsqueda de influencers con imagen.

## 6) Notas

- La primera inferencia de matching puede tardar más por calentamiento del modelo.
- Ejecuciones posteriores en el mismo proceso suelen ser más rápidas.
- Si el navegador muestra estilos viejos, usar hard refresh (`Ctrl+F5`).

## 7) Legacy

Siguen existiendo scripts/apps legacy (`app_server.py`, `hackathon/app.py`) para referencia, pero el flujo principal actual es Django.


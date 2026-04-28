# Scuffers Creator Match AI (Local Setup)

This project has two local apps:

- `app_server.py` -> FastAPI + Gradio (`/ui`) for product-to-creator style matching.
- `hackathon/app.py` -> Streamlit Control Tower dashboard (includes button to open matching UI).

If Railway fails, this guide lets anyone run the demo on their own machine in a few minutes.

## 1) Requirements

- Python 3.10+ (recommended 3.10/3.11)
- Windows / macOS / Linux
- Internet on first run (OpenCLIP weights may download)

## 2) Install

From repo root:

```bash
python -m venv .venv
```

Activate environment:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) Run the apps (two terminals)

Open two terminals at repo root with the same virtual environment active.

### Terminal A: Matching API + UI

```bash
python -m uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Matching UI: [http://127.0.0.1:8000/ui?v=mini](http://127.0.0.1:8000/ui?v=mini)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Terminal B: Dashboard

```bash
streamlit run hackathon/app.py --server.port 8501
```

Open:

- Dashboard: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## 4) Demo flow

1. Open Dashboard.
2. Go to `Creators Growth`.
3. Click `Ver influencers` (opens matching UI).
4. In matching UI:
   - Upload product image
   - Choose country (optional)
   - Toggle `Business` or `Style only`
   - Click `Find Influencers`
5. Review:
   - Influencer clusters plot
   - Similarity bar chart
   - Top 3 creators with image + username

## 5) Known behavior

- First inference can be slow on CPU (ViT-H model warm-up/download).
- Next runs in same process are faster (model cache in memory).

## 6) Troubleshooting

- **Browser shows old UI:** open `http://127.0.0.1:8000/ui?v=mini` and hard refresh (`Ctrl+F5`).
- **Port already in use:** change ports (`8001`, `8502`) or stop existing process.
- **Do not open `http://0.0.0.0:8000` in browser:** use `127.0.0.1` or `localhost`.
- **No matches / errors:** verify local data files exist under `data/influencers/`.

## 7) Optional: run only matching UI

If you only need the AI matcher (without dashboard):

```bash
python -m uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://127.0.0.1:8000/ui?v=mini](http://127.0.0.1:8000/ui?v=mini).


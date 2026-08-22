# Crop Doctor — Web Version

Real MobileNetV2-based crop disease detection (same 27-class model your
Flutter app uses), as a web app. Four files, as requested:

```
frontend/index.html   → the whole UI
frontend/style.css    → all styling
frontend/app.js       → all frontend logic (calls the backend API)
backend/app.py        → Flask API: real tflite inference + Neon history
```
(plus small config files: `backend/requirements.txt`, `backend/render.yaml`,
`frontend/netlify.toml`, and the model file `backend/crop_doctor_model.tflite`
copied from your Flutter project's `assets/ml/`.)

## Architecture

```
Netlify (index.html/style.css/app.js)
        │  fetch()
        ▼
Render (app.py — Flask + real MobileNetV2 tflite model)
        │  psycopg2
        ▼
Neon (Postgres — analysis_history table)
```

Diagnosis never depends on the database or on weather — if Neon or the
weather API is unreachable, image analysis still works and history just
falls back to the browser's localStorage.

## 1. Deploy the backend on Render

1. Push the `backend/` folder to a GitHub repo (or the whole project —
   Render just needs a root that contains `app.py`).
2. On Render: **New → Web Service** → connect the repo.
   - Root directory: `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Plan: Free is fine to start.
3. Add environment variables (Render dashboard → Environment):
   - `DATABASE_URL` — your Neon connection string (step 2 below)
   - `FRONTEND_ORIGIN` — your Netlify URL, e.g. `https://crop-doctor.netlify.app`
     (use `*` temporarily while testing, then lock it down)
4. Deploy. Once live, check `https://<your-service>.onrender.com/api/health`
   — it should show `"model_loaded": true`.

**Note on `tflite-runtime`:** it doesn't publish wheels for every Python
version. If Render's build fails installing it, open `requirements.txt`
and replace the `tflite-runtime` line with `tensorflow-cpu==2.16.1` — the
code already falls back to `tensorflow.lite.Interpreter` automatically,
no code changes needed. It's a bigger install (may need Render's Starter
plan instead of Free for enough RAM/build time).

## 2. Set up the database on Neon

1. Create a free project at neon.tech.
2. Copy the connection string it gives you (starts `postgresql://...`,
   includes `?sslmode=require`).
3. Paste it into Render as `DATABASE_URL` (step above). The backend
   creates its own `analysis_history` table automatically on first boot —
   no manual SQL needed.

## 3. Deploy the frontend on Netlify

1. Push the `frontend/` folder to GitHub (or drag-and-drop the folder
   directly onto Netlify's dashboard — no build step required, it's
   static files).
2. On Netlify: **Add new site → Deploy manually** (drag & drop) or
   **Import from Git**, with:
   - Base directory: `frontend`
   - Build command: *(leave empty)*
   - Publish directory: `frontend` (or `.` if the repo root *is* the
     frontend folder)
3. Once live, open the site → **Settings tab** in the app → paste your
   Render URL (e.g. `https://crop-doctor-api.onrender.com`) into
   **Backend API URL** → Save. It checks the connection immediately.

## 4. Connect Ollama for the AI Assistant (optional, local)

The AI Assistant works with basic canned answers out of the box. For full
conversational answers:

```bash
# On your own machine or a server you control:
ollama pull llama3
ollama serve
```

Then in the web app's **Settings** tab, set **Ollama AI Assistant URL** to
wherever Ollama is reachable (e.g. `http://localhost:11434` if you're
running the site locally too, or a tunneled/public URL if Ollama runs on
a separate machine from the browser). The assistant always receives the
real ML diagnosis as context and is instructed never to override it —
Ollama only explains, it never re-diagnoses.

## Limitations / what needs internet

- **Works without internet:** the actual disease diagnosis (frontend +
  backend + model are all yours), viewing symptoms/recommendations/prevention.
- **Needs internet:** loading the site itself, the backend API call, Neon
  history storage (falls back to localStorage if unreachable), the
  Wikipedia "Learn More" links, weather/risk data (Open-Meteo, free, no
  key — shows "Weather intelligence unavailable" if offline), and the
  Ollama AI Assistant if you host it remotely.

## Files created
- `backend/app.py`, `backend/requirements.txt`, `backend/render.yaml`, `backend/crop_doctor_model.tflite`
- `frontend/index.html`, `frontend/style.css`, `frontend/app.js`, `frontend/netlify.toml`

## Environment variables needed
- Render: `DATABASE_URL` (Neon connection string), `FRONTEND_ORIGIN` (your Netlify URL)
- Frontend: no env vars — the backend URL is set in-app under Settings (stored in the browser's localStorage)

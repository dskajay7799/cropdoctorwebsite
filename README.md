# Crop Doctor

AI-assisted crop leaf disease diagnosis. Upload a leaf photo and get a
crop + condition prediction from a 39-class MobileNetV2 model, plus
structured, real-world diagnosis information (symptoms, likely cause,
recommended action, prevention, severity).

## What changed in this upgrade

- **Model**: `crop_doctor_model.tflite` (27 classes) → `CropDoctor_39Class_MobileNetV2.tflite`
  (39 classes, ~84.9% validation accuracy, trained on 1,560 images).
- **Class mapping**: driven entirely by `class_names.json` (no hardcoded order).
- **Crop selection removed**: the old UI made you pick a crop before uploading.
  The new model detects the crop itself, so this step is gone — one photo,
  one result. Fewer clicks, less room for user error (picking the wrong crop
  used to trigger a false "mismatch" error).
- **Diagnosis depth**: every one of the 39 classes now has curated
  cause / symptoms / recommended action / prevention / severity content,
  not just a bare label.
- **Confidence handling**: three tiers (high ≥75%, moderate ≥45%, low <45%)
  with different messaging — low confidence is clearly flagged as uncertain
  rather than presented as fact.
- **Frontend redesign**: removed the emoji/icon-heavy UI, tightened
  typography and spacing, and restructured the result view into a clear
  hierarchy (crop + condition → confidence → explanation → symptoms →
  action → prevention → disclaimer).
- **Efficiency**: the TFLite interpreter is loaded once at process start
  and reused for every request (not reloaded per call). The `.keras` file
  is not loaded in production — it's kept only as a training backup.
- **Error handling**: invalid/missing images, unreadable files, oversized
  uploads (>12MB), and model-loading failures all return clean JSON errors
  instead of leaking a Python traceback.

## Project structure

```
CropDoctor/
├── app.py                    # Flask API (backend)
├── requirements.txt
├── render.yaml
├── class_names.json           # 39-class label order (source of truth)
├── CropDoctor_39Class_MobileNetV2.tflite   # ← add this file yourself
├── index.html                 # Frontend (HTML + CSS + JS, single file)
├── netlify.toml
└── README.md
```

> **Important:** `CropDoctor_39Class_MobileNetV2.tflite` couldn't be carried
> over into this project export, so it isn't included here. Just drop it
> into this same folder, next to `app.py` — the code looks for it at
> `CropDoctor_39Class_MobileNetV2.tflite` relative to `app.py`, so no path
> changes are needed. The `.keras` file is optional and not required at
> runtime; keep it separately as a training backup if you want.

## How the model is used

- Input: RGB image, resized to 224×224, cast to `float32`.
- **Pixel scaling assumption**: pixels are scaled to `[0, 1]` (divided by
  255). This matches the approach used in the previous model. If real-world
  accuracy looks noticeably worse than the ~84.9% validation accuracy you
  saw during training, the most likely cause is a preprocessing mismatch —
  try `[-1, 1]` scaling instead (`(pixel - 127.5) / 127.5`, the standard
  `tf.keras.applications.mobilenet_v2.preprocess_input`). This is controlled
  by `IMG_MEAN` / `IMG_STD` near the top of `backend/app.py`.
- Output: 39 probabilities, mapped to `class_names.json` in order — the
  file is loaded at runtime, not hardcoded, so it will always stay in sync
  if you retrain with a different class order.
- Each class name is split into crop + condition (e.g.
  `Bell_Pepper_Bacterial_Spot` → crop "Bell Pepper", condition
  "Bacterial Spot"; `*_Healthy` classes are treated as the healthy case).

## Backend setup (local)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# place CropDoctor_39Class_MobileNetV2.tflite in this same folder
python app.py
```

Runs on `http://localhost:5000` by default (`PORT` env var to override).

### Environment variables

| Variable          | Required | Purpose                                              |
|--------------------|----------|-------------------------------------------------------|
| `DATABASE_URL`      | No       | Neon/Postgres connection string for persistent history. Falls back to in-memory storage if unset. |
| `FRONTEND_ORIGIN`   | No       | CORS origin allowed to call the API (defaults to `*`). Set this to your Netlify URL in production. |
| `GROQ_API_KEY`      | No       | Enables the `/api/chat` AI Assistant. Without it, the assistant tab still works using simple offline canned answers based on the diagnosis. |
| `GROQ_MODEL`        | No       | Groq model id (defaults to `openai/gpt-oss-20b`). |

## Frontend setup (local)

`index.html` is self-contained — open it directly in a browser, or serve it
with any static server:

```bash
python -m http.server 8080
```

Set the backend URL in the app's **Settings** tab (defaults to
`https://cropdoctorwebsite.onrender.com` — change this to your own Render
URL, or to `http://localhost:5000` for local testing).

## Deployment

- **Backend → Render**: connect this folder as a Web Service (or split
  `app.py`, `requirements.txt`, `render.yaml`, `class_names.json`, and the
  `.tflite` file into their own repo/folder if you prefer separate repos).
  `render.yaml` is already set up (`gunicorn app:app`). Set `DATABASE_URL`,
  `FRONTEND_ORIGIN`, and optionally `GROQ_API_KEY` in the Render dashboard.
  Make sure `CropDoctor_39Class_MobileNetV2.tflite` and `class_names.json`
  are committed to the repo alongside `app.py`.
- **Frontend → Netlify**: deploy `index.html` (and `netlify.toml`) to your
  existing site at `boisterous-froyo-7ebad8.netlify.app`.

## API reference

### `GET /api/health`
```json
{ "status": "ok", "model_loaded": true, "num_classes": 39, "database_connected": true }
```

### `GET /api/classes`
Returns the raw list of 39 class strings from `class_names.json`.

### `POST /api/analyze`
`multipart/form-data` with a single `image` file field.

Success response:
```json
{
  "status": "success",
  "crop": "Tomato",
  "condition": "Early Blight",
  "is_healthy": false,
  "confidence": 87.3,
  "confidence_tier": "high",
  "severity": "Attention recommended",
  "explanation": "The image shows signs consistent with early blight on tomato.",
  "cause": "The fungus Alternaria solani, favored by warm temperatures and periods of leaf wetness.",
  "symptoms": ["...", "...", "..."],
  "action": ["...", "...", "..."],
  "prevention": ["...", "...", "..."],
  "disclaimer": "This is an AI-assisted assessment, not a guaranteed laboratory diagnosis. ..."
}
```
Other statuses: `image_unreadable` (400), `model_unavailable` (503), and
generic `error` (400/500) for missing files or unexpected failures — none
of these leak internal Python error text to the client.

### `GET/POST/DELETE /api/history`, `DELETE /api/history/<id>`
Same shape as before, saving `{crop, condition, confidence, severity, status}`.

### `POST /api/chat`
`{"question": "...", "context": <last analyze() response, optional>}` →
`{"status": "success", "answer": "..."}`. Requires `GROQ_API_KEY`; returns
`503` with a clear message if not configured (frontend falls back to
canned offline answers in that case).

## Confidence thresholds (documented, not hidden)

Defined in `backend/app.py`:

- **High** (≥ 75%): shown as a normal result.
- **Moderate** (≥ 45%, < 75%): result shown, with a note recommending
  verification before acting on it.
- **Low** (< 45%): result shown but clearly flagged as uncertain, with a
  suggestion to retake the photo or get expert confirmation.

## What was intentionally left unchanged

- The Neon/Postgres history feature, the Groq-powered assistant chat, and
  the overall Flask/vanilla-JS architecture were kept as-is — only the
  model, diagnosis logic, and UI were changed, per the request to avoid
  breaking working parts.
- `app.js` (a standalone copy of the old inline script) was not wired into
  `index.html` in the original project — the live site's actual JS lives
  inline inside `index.html`. This upgrade continues that pattern; `app.js`
  is no longer needed and can be deleted from the repo if you don't use it
  elsewhere.

## Responsible use

This tool provides an AI-assisted assessment, not a laboratory diagnosis.
For high-value crops, ambiguous symptoms, or low-confidence results, confirm
with a local agricultural extension office or expert before taking action.

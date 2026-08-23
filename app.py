"""
Crop Doctor — Backend API
==========================
Flask API that serves the REAL Crop Doctor MobileNetV2 model
(crop_doctor_model.tflite, 27 classes, PlantVillage-trained) and stores
analysis history in a Neon (serverless Postgres) database.

Deploy target: Render (Web Service)
Database:      Neon Postgres (set DATABASE_URL env var)
Frontend:      Netlify (calls this API, set FRONTEND_ORIGIN env var for CORS)

Endpoints
---------
GET  /api/health              -> {"status": "ok", "model_loaded": true/false}
POST /api/analyze             -> multipart/form-data: image=<file>, crop=<crop_id>
                                  returns diagnosis JSON (never fakes a result)
GET  /api/history             -> list of saved analyses (most recent first)
POST /api/history             -> save an analysis record {crop, disease, confidence, severity, status}
DELETE /api/history/<id>      -> delete one record
DELETE /api/history           -> clear all history

Nothing here invents an ML prediction. If the model file is missing or
fails to load, /api/analyze returns a clear "model unavailable" error
instead of a fabricated diagnosis.
"""

import io
import os
import json
import uuid
import datetime

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# tflite runtime: prefer the lightweight tflite-runtime package (small,
# Render-friendly). Fall back to full tensorflow if that's what's installed.
# ---------------------------------------------------------------------------
import tensorflow as tf

Interpreter = tf.lite.Interpreter

# ---------------------------------------------------------------------------
# Optional Postgres (Neon) support. If DATABASE_URL isn't set, history
# endpoints fall back to an in-memory list so the API still runs locally
# without a database configured.
# ---------------------------------------------------------------------------
try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_doctor_model.tflite")
DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

# Free AI Assistant backend (Groq — no cost, no credit card). Get a free
# key at https://console.groq.com/keys and set it as GROQ_API_KEY in
# Render's environment variables. The key lives only on the server —
# it is never sent to or visible in the frontend.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONTEND_ORIGIN}})

# ---------------------------------------------------------------------------
# Model labels — EXACT order used during training. Do not reorder.
# ---------------------------------------------------------------------------
LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

MODEL_CROP_TO_APP_CROP_ID = {
    "Apple": "apple",
    "Corn_(maize)": "maize",
    "Grape": "grape",
    "Pepper,_bell": "pepper",
    "Potato": "potato",
    "Tomato": "tomato",
}

MODEL_SUPPORTED_CROP_IDS = {"apple", "maize", "grape", "pepper", "potato", "tomato"}

# All crops shown in the UI. Crops with model_supported=False are still
# selectable (per product spec) but diagnosis is refused honestly instead
# of being faked, since the installed model has no classes for them.
ALL_CROPS = [
    {"id": "rice", "label": "Rice", "emoji": "🌾", "model_supported": False},
    {"id": "wheat", "label": "Wheat", "emoji": "🌿", "model_supported": False},
    {"id": "maize", "label": "Maize / Corn", "emoji": "🌽", "model_supported": True},
    {"id": "tomato", "label": "Tomato", "emoji": "🍅", "model_supported": True},
    {"id": "potato", "label": "Potato", "emoji": "🥔", "model_supported": True},
    {"id": "apple", "label": "Apple", "emoji": "🍎", "model_supported": True},
    {"id": "grape", "label": "Grape", "emoji": "🍇", "model_supported": True},
    {"id": "pepper", "label": "Pepper", "emoji": "🌶️", "model_supported": True},
    {"id": "soybean", "label": "Soybean", "emoji": "🫘", "model_supported": False},
    {"id": "banana", "label": "Banana", "emoji": "🍌", "model_supported": False},
    {"id": "mango", "label": "Mango", "emoji": "🥭", "model_supported": False},
    {"id": "groundnut", "label": "Groundnut", "emoji": "🥜", "model_supported": False},
    {"id": "onion", "label": "Onion", "emoji": "🧅", "model_supported": False},
]

# ---------------------------------------------------------------------------
# Curated, verified disease knowledge base (ported from the Flutter app's
# disease_knowledge_base.dart). Keyed by (raw model crop segment, disease).
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    ("Apple", "Apple_scab"): dict(
        symptoms=["Olive-green to dark brown velvety spots on leaves",
                   "Corky, scab-like lesions on fruit skin",
                   "Premature leaf yellowing and drop"],
        recommendations=["Remove and destroy fallen leaves to reduce fungal spores",
                          "Apply a recommended fungicide at green-tip and repeat per label interval",
                          "Prune to improve air circulation through the canopy"],
        prevention=["Choose scab-resistant apple varieties where possible",
                     "Avoid overhead irrigation that keeps foliage wet",
                     "Rake and dispose of leaf litter each autumn"],
    ),
    ("Apple", "Black_rot"): dict(
        symptoms=["Purple-bordered leaf spots (\"frog-eye leaf spot\")",
                   "Rotting, mummified fruit with concentric rings",
                   "Sunken, reddish-brown cankers on branches"],
        recommendations=["Prune out and destroy cankered wood and mummified fruit",
                          "Apply fungicide during bloom through early summer per label",
                          "Remove nearby dead or diseased wood that can harbor spores"],
        prevention=["Sanitize pruning tools between cuts",
                     "Avoid wounding bark and fruit during harvest",
                     "Maintain tree vigor with balanced fertilization"],
    ),
    ("Apple", "Cedar_apple_rust"): dict(
        symptoms=["Bright yellow-orange spots on upper leaf surface",
                   "Small raised cups or tubes on the underside of leaves",
                   "Distorted or spotted fruit in severe cases"],
        recommendations=["Apply a protective fungicide from pink bud through early summer",
                          "Remove nearby juniper/cedar hosts within a few hundred meters if feasible",
                          "Rake and destroy fallen infected leaves"],
        prevention=["Plant rust-resistant apple varieties",
                     "Avoid planting apples near ornamental junipers",
                     "Monitor closely in wet spring weather, which favors spread"],
    ),
    ("Corn_(maize)", "Cercospora_leaf_spot Gray_leaf_spot"): dict(
        symptoms=["Small, rectangular tan-to-gray lesions bound by leaf veins",
                   "Lesions merge in severe infections, blighting whole leaves",
                   "Symptoms usually start on lower leaves and move upward"],
        recommendations=["Apply a foliar fungicide if disease appears before tasseling",
                          "Rotate with a non-host crop for at least one season",
                          "Select resistant hybrids for future plantings"],
        prevention=["Practice residue management/tillage to reduce overwintering spores",
                     "Avoid continuous corn-on-corn planting in the same field",
                     "Ensure adequate plant spacing for airflow"],
    ),
    ("Corn_(maize)", "Common_rust_"): dict(
        symptoms=["Small, cinnamon-brown, powdery pustules on both leaf surfaces",
                   "Pustules rupture the leaf epidermis, releasing rust-colored spores",
                   "Heavily infected leaves may yellow and die prematurely"],
        recommendations=["Apply fungicide if pustules appear early and conditions stay cool/humid",
                          "Favor rust-resistant hybrids in future seasons",
                          "Monitor fields regularly during cool, humid weather"],
        prevention=["Plant resistant hybrids where common rust is a recurring issue",
                     "Avoid excessive nitrogen that promotes dense, humid canopies",
                     "Scout early since rust spreads quickly in favorable weather"],
    ),
    ("Corn_(maize)", "Northern_Leaf_Blight"): dict(
        symptoms=["Long, cigar-shaped gray-green to tan lesions on leaves",
                   "Lesions can span several centimeters and merge together",
                   "Severe infection causes premature leaf death and yield loss"],
        recommendations=["Apply a labeled fungicide, especially at or before tasseling",
                          "Rotate crops away from corn for at least one year",
                          "Till under crop residue where the fungus overwinters"],
        prevention=["Plant hybrids with genetic resistance to Northern Leaf Blight",
                     "Avoid dense planting that keeps the canopy humid",
                     "Scout fields regularly, especially after warm, wet weather"],
    ),
    ("Grape", "Black_rot"): dict(
        symptoms=["Small tan spots with dark borders on leaves",
                   "Fruit shrivels into hard, black \"mummies\"",
                   "Reddish-brown lesions can appear on shoots and tendrils"],
        recommendations=["Remove and destroy mummified berries and infected leaves/canes",
                          "Apply fungicide from early shoot growth through veraison per label",
                          "Improve canopy airflow through timely pruning"],
        prevention=["Prune out infected wood during dormancy",
                     "Avoid working in the vineyard when foliage is wet",
                     "Choose sites with good air movement and sun exposure"],
    ),
    ("Grape", "Esca_(Black_Measles)"): dict(
        symptoms=["\"Tiger-stripe\" interveinal yellowing/browning on leaves",
                   "Dark spotting on berries in severe cases",
                   "Internal wood streaking and dieback of cordons/trunk"],
        recommendations=["Remove and destroy severely affected vines/wood to slow spread",
                          "Protect large pruning wounds with a wound sealant",
                          "Avoid pruning during wet weather, when infection risk is highest"],
        prevention=["Use delayed or double pruning to reduce wound exposure time",
                     "Sanitize pruning tools between vines",
                     "Maintain overall vine health to improve tolerance"],
    ),
    ("Grape", "Leaf_blight_(Isariopsis_Leaf_Spot)"): dict(
        symptoms=["Angular brown-to-black spots on leaves, often with a yellow halo",
                   "Spots may merge, leading to premature leaf drop",
                   "Reduced vine vigor with repeated defoliation"],
        recommendations=["Apply a protective fungicide during the growing season per label",
                          "Remove fallen, infected leaves at season end",
                          "Improve canopy ventilation through leaf pulling/pruning"],
        prevention=["Avoid overhead irrigation that prolongs leaf wetness",
                     "Maintain good weed control to improve airflow near the ground",
                     "Monitor closely during warm, humid periods"],
    ),
    ("Pepper,_bell", "Bacterial_spot"): dict(
        symptoms=["Small, water-soaked spots on leaves that turn brown and scab-like",
                   "Raised, rough spots on fruit surface",
                   "Leaf yellowing and drop in severe cases"],
        recommendations=["Remove and destroy severely infected plants/leaves",
                          "Apply a copper-based bactericide early, per local label guidance",
                          "Avoid working among wet plants to reduce spread"],
        prevention=["Use certified disease-free seed and transplants",
                     "Rotate away from peppers/tomatoes for 1-2 seasons",
                     "Avoid overhead watering; water at the base instead"],
    ),
    ("Potato", "Early_blight"): dict(
        symptoms=["Dark, concentric \"target-ring\" spots on older leaves first",
                   "Yellowing tissue surrounding leaf spots",
                   "Lesions can also appear on stems and tubers"],
        recommendations=["Remove and destroy heavily infected lower leaves",
                          "Apply a labeled fungicide on a preventive schedule in humid weather",
                          "Maintain balanced fertility — stressed plants are more susceptible"],
        prevention=["Rotate potatoes with non-host crops for 2+ years",
                     "Space plants for good airflow and faster leaf drying",
                     "Avoid overhead irrigation late in the day"],
    ),
    ("Potato", "Late_blight"): dict(
        symptoms=["Water-soaked, pale-to-dark green lesions that turn brown/black quickly",
                   "White fungal growth on the underside of leaves in humid conditions",
                   "Firm, dark, granular rot on tubers"],
        recommendations=["Act quickly: remove and destroy infected foliage/plants",
                          "Apply a labeled fungicide immediately — this disease spreads fast",
                          "Avoid irrigating or harvesting in wet conditions once detected"],
        prevention=["Plant certified, disease-free seed potatoes",
                     "Destroy volunteer potato plants and cull piles",
                     "Monitor closely during cool, wet weather, which favors outbreaks"],
    ),
    ("Tomato", "Bacterial_spot"): dict(
        symptoms=["Small, water-soaked, greasy-looking leaf spots that turn dark",
                   "Raised, scabby spots on green fruit",
                   "Leaf yellowing and defoliation in severe cases"],
        recommendations=["Remove and destroy severely infected leaves/plants",
                          "Apply a copper-based bactericide per local label guidance",
                          "Avoid handling wet plants to limit spread between them"],
        prevention=["Use certified disease-free seed and transplants",
                     "Rotate away from tomatoes/peppers for 1-2 seasons",
                     "Water at the base of plants, not overhead"],
    ),
    ("Tomato", "Early_blight"): dict(
        symptoms=["Dark, concentric \"target-ring\" spots on older/lower leaves",
                   "Yellow halo surrounding leaf spots",
                   "Dark, leathery lesions can form near the fruit stem"],
        recommendations=["Remove and destroy infected lower leaves promptly",
                          "Apply a labeled fungicide on a preventive schedule in humid weather",
                          "Stake or cage plants to keep foliage off the soil"],
        prevention=["Rotate tomatoes with non-host crops for 2+ years",
                     "Mulch to reduce soil splashing spores onto leaves",
                     "Avoid overhead irrigation late in the day"],
    ),
    ("Tomato", "Late_blight"): dict(
        symptoms=["Large, water-soaked, pale-green to brown blotches on leaves",
                   "White, fuzzy fungal growth on leaf undersides in humid weather",
                   "Firm, dark, greasy-looking rot on fruit"],
        recommendations=["Act quickly: remove and destroy infected foliage/plants",
                          "Apply a labeled fungicide immediately — this disease spreads fast",
                          "Avoid overhead watering once detected"],
        prevention=["Space and stake plants for good airflow",
                     "Avoid planting near infected potato fields",
                     "Monitor closely during cool, wet, humid weather"],
    ),
    ("Tomato", "Leaf_Mold"): dict(
        symptoms=["Pale green-to-yellow spots on the upper leaf surface",
                   "Olive-green to grayish-brown fuzzy mold on the underside",
                   "Common in humid greenhouse/covered growing conditions"],
        recommendations=["Improve ventilation and reduce humidity around plants",
                          "Remove and destroy heavily infected leaves",
                          "Apply a labeled fungicide if conditions stay humid"],
        prevention=["Space plants and prune to improve air circulation",
                     "Water at the base of plants, avoiding wet foliage",
                     "Choose resistant tomato varieties where available"],
    ),
    ("Tomato", "Septoria_leaf_spot"): dict(
        symptoms=["Numerous small, circular spots with dark borders and gray centers",
                   "Tiny black specks (fungal fruiting bodies) visible in spot centers",
                   "Lower leaves affected first, often causing yellowing and drop"],
        recommendations=["Remove and destroy infected lower leaves promptly",
                          "Apply a labeled fungicide on a preventive schedule",
                          "Avoid working among wet plants to limit spread"],
        prevention=["Rotate tomatoes with non-host crops for at least one season",
                     "Mulch to reduce soil splash onto lower leaves",
                     "Stake or cage plants to keep foliage off the ground"],
    ),
    ("Tomato", "Spider_mites Two-spotted_spider_mite"): dict(
        symptoms=["Fine yellow/white stippling on leaf surfaces",
                   "Fine webbing on leaves and stems in heavy infestations",
                   "Leaves may bronze, dry out, and drop in severe cases"],
        recommendations=["Rinse plants with a strong water spray to dislodge mites",
                          "Apply an appropriate miticide or insecticidal soap if severe",
                          "Introduce or conserve natural predators where feasible"],
        prevention=["Avoid drought-stressing plants, which favors mite outbreaks",
                     "Monitor closely during hot, dry weather",
                     "Remove heavily infested leaves early"],
    ),
    ("Tomato", "Target_Spot"): dict(
        symptoms=["Brown, concentric-ringed spots on leaves, stems and fruit",
                   "Spots may merge, causing large blighted areas",
                   "Can lead to significant defoliation in humid conditions"],
        recommendations=["Remove and destroy infected leaves/debris",
                          "Apply a labeled fungicide on a preventive schedule in humid weather",
                          "Improve airflow through staking and pruning"],
        prevention=["Rotate with non-host crops where possible",
                     "Avoid overhead irrigation late in the day",
                     "Space plants adequately for airflow"],
    ),
    ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"): dict(
        symptoms=["Upward curling and yellowing of leaflet margins",
                   "Stunted plant growth and reduced fruit set",
                   "Spread primarily by whiteflies"],
        recommendations=["Remove and destroy infected plants to reduce virus reservoirs",
                          "Control whitefly populations with appropriate measures",
                          "There is no cure once a plant is infected — focus on prevention"],
        prevention=["Use whitefly-resistant or tolerant tomato varieties",
                     "Use reflective mulch or fine mesh netting to deter whiteflies",
                     "Remove nearby weeds that can host whiteflies/virus"],
    ),
    ("Tomato", "Tomato_mosaic_virus"): dict(
        symptoms=["Light and dark green mottling/mosaic pattern on leaves",
                   "Leaf distortion, curling, or fern-like narrowing",
                   "Stunted growth and reduced, sometimes mottled, fruit"],
        recommendations=["Remove and destroy infected plants — there is no cure",
                          "Wash hands and disinfect tools after handling infected plants",
                          "Avoid tobacco use near plants; the virus can be transmitted this way"],
        prevention=["Use certified virus-free seed and resistant varieties",
                     "Disinfect tools and hands between plants when pruning/staking",
                     "Control weeds that may harbor the virus"],
    ),
}

# ---------------------------------------------------------------------------
# Model loading (once, at process start)
# ---------------------------------------------------------------------------
_interpreter = None
_input_details = None
_output_details = None


def load_model():
    global _interpreter, _input_details, _output_details

    if _interpreter is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("MODEL FILE NOT FOUND:", MODEL_PATH)
        return

    try:
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()

        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        print("========================================")
        print("CROP DOCTOR MODEL LOADED")
        print("MODEL PATH:", MODEL_PATH)
        print("INPUT DETAILS:", _input_details)
        print("OUTPUT DETAILS:", _output_details)
        print("========================================")

    except Exception as e:
        print("MODEL LOAD ERROR:", repr(e))
        _interpreter = None
        _input_details = None
        _output_details = None


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "moderate"
    return "low"


def severity_from_tier(tier: str, is_healthy: bool) -> str:
    if is_healthy:
        return "none"
    return {"high": "High", "moderate": "Moderate", "low": "Low"}[tier]


def run_inference(image_bytes: bytes):
    """
    Runs the real TFLite model.

    The preprocessing is determined from the actual TFLite
    input tensor instead of assuming a fixed dtype.
    """

    if _interpreter is None:
        raise RuntimeError("TFLite model is not loaded.")

    if not _input_details or not _output_details:
        raise RuntimeError("TFLite model tensor information is unavailable.")

    # ---------------------------------------------------------
    # 1. READ IMAGE
    # ---------------------------------------------------------
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Invalid image file: {e}")

    # ---------------------------------------------------------
    # 2. READ ACTUAL MODEL INPUT SPECIFICATION
    # ---------------------------------------------------------
    input_info = _input_details[0]

    input_index = input_info["index"]
    input_shape = input_info["shape"]
    input_dtype = input_info["dtype"]

    print("MODEL INPUT SHAPE:", input_shape)
    print("MODEL INPUT DTYPE:", input_dtype)
    print("MODEL INPUT QUANTIZATION:", input_info.get("quantization"))

    # Expected shape should normally be:
    # [1, 224, 224, 3]

    if len(input_shape) != 4:
        raise RuntimeError(
            f"Unexpected model input shape: {input_shape}"
        )

    height = int(input_shape[1])
    width = int(input_shape[2])
    channels = int(input_shape[3])

    if channels != 3:
        raise RuntimeError(
            f"Model expects {channels} channels instead of RGB 3 channels."
        )

    # ---------------------------------------------------------
    # 3. RESIZE TO ACTUAL MODEL SIZE
    # ---------------------------------------------------------
    img = img.resize((width, height))

    arr = np.asarray(img)

    # ---------------------------------------------------------
    # 4. PREPROCESS ACCORDING TO MODEL DTYPE
    # ---------------------------------------------------------
    if input_dtype == np.float32:

        # Float32 MobileNetV2 model.
        # This is correct ONLY if the training pipeline
        # used pixel values in the 0-1 range.
        arr = arr.astype(np.float32) / 255.0

    elif input_dtype == np.uint8:

        # UINT8 quantized model.
        arr = arr.astype(np.uint8)

    elif input_dtype == np.int8:

        # INT8 quantized model.
        scale, zero_point = input_info.get("quantization", (0.0, 0))

        if scale == 0:
            raise RuntimeError(
                "Invalid INT8 quantization scale."
            )

        arr = arr.astype(np.float32)

        arr = np.round(
            arr / scale + zero_point
        )

        arr = np.clip(
            arr,
            -128,
            127
        ).astype(np.int8)

    else:
        raise RuntimeError(
            f"Unsupported model input dtype: {input_dtype}"
        )

    # ---------------------------------------------------------
    # 5. ADD BATCH DIMENSION
    # ---------------------------------------------------------
    arr = np.expand_dims(arr, axis=0)

    print("FINAL INPUT SHAPE:", arr.shape)
    print("FINAL INPUT DTYPE:", arr.dtype)

    # ---------------------------------------------------------
    # 6. VERIFY SHAPE BEFORE SENDING TO TFLITE
    # ---------------------------------------------------------
    expected_shape = tuple(input_shape)
    actual_shape = tuple(arr.shape)

    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Input shape mismatch. "
            f"Model expects {expected_shape}, "
            f"but received {actual_shape}."
        )

    # ---------------------------------------------------------
    # 7. RUN TFLITE
    # ---------------------------------------------------------
    _interpreter.set_tensor(
        input_index,
        arr
    )

    _interpreter.invoke()

    # ---------------------------------------------------------
    # 8. GET MODEL OUTPUT
    # ---------------------------------------------------------
    output_info = _output_details[0]

    output = _interpreter.get_tensor(
        output_info["index"]
    )

    output = np.asarray(output)

    print("RAW OUTPUT SHAPE:", output.shape)
    print("RAW OUTPUT DTYPE:", output.dtype)
    print("RAW OUTPUT:", output)

    # Remove batch dimension
    output = output[0]

    # ---------------------------------------------------------
    # 9. HANDLE QUANTIZED OUTPUT
    # ---------------------------------------------------------
    output_dtype = output_info["dtype"]

    if output_dtype == np.uint8 or output_dtype == np.int8:

        scale, zero_point = output_info.get(
            "quantization",
            (0.0, 0)
        )

        if scale != 0:
            output = (
                output.astype(np.float32) - zero_point
            ) * scale
        else:
            output = output.astype(np.float32)

    else:
        output = output.astype(np.float32)

    # Flatten if necessary
    output = output.flatten()

    # ---------------------------------------------------------
    # 10. VERIFY 27 OUTPUT CLASSES
    # ---------------------------------------------------------
    if len(output) != len(LABELS):
        raise RuntimeError(
            f"Model output has {len(output)} values, "
            f"but backend has {len(LABELS)} labels."
        )

    # ---------------------------------------------------------
    # 11. ENSURE WE HAVE PROBABILITIES
    # ---------------------------------------------------------
    total = float(np.sum(output))

    if (
        np.min(output) < 0
        or np.max(output) > 1
        or not np.isclose(total, 1.0, atol=0.05)
    ):
        # Treat output as logits and apply softmax.
        shifted = output - np.max(output)

        exp_output = np.exp(shifted)

        output = exp_output / np.sum(exp_output)

    print("FINAL PROBABILITIES:", output)
    print("PROBABILITY SUM:", float(np.sum(output)))

    return output

# ---------------------------------------------------------------------------
# Database (Neon Postgres) — analysis history
# ---------------------------------------------------------------------------
_memory_history = []  # fallback store used only when DATABASE_URL isn't set


def get_db():
    if not DATABASE_URL or psycopg2 is None:
        return None
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    return conn


def init_db():
    conn = get_db()
    if conn is None:
        return
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                crop TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
    conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _interpreter is not None,
        "database_connected": DATABASE_URL is not None,
    })


@app.route("/api/crops", methods=["GET"])
def crops():
    return jsonify(ALL_CROPS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    crop_id = request.form.get("crop", "")
    image_file = request.files.get("image")

    if not crop_id:
        return jsonify({"status": "error", "message": "No crop selected."}), 400
    if image_file is None:
        return jsonify({"status": "error", "message": "No image was uploaded."}), 400

    # Guard 1: crop the model has no classes for at all.
    if crop_id not in MODEL_SUPPORTED_CROP_IDS:
        return jsonify({
            "status": "crop_not_supported",
            "message": "This crop isn't supported by the trained model yet.",
            "selected_crop": crop_id,
        })

    if _interpreter is None:
        load_model()
    if _interpreter is None:
        return jsonify({
            "status": "model_unavailable",
            "message": "AI model is being connected. Please try again shortly.",
        }), 503

    image_bytes = image_file.read()

    try:
        probabilities = run_inference(image_bytes)
    except Exception as e:
        print("INFERENCE ERROR:", str(e))
        return jsonify({
            "status": "inference_error",
            "message": "The AI model could not analyze this image.",
            "error": str(e)
        }), 500

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    print("========================================")
    print("PREDICTION")
    print("BEST INDEX:", best_index)
    print("LABEL:", LABELS[best_index])
    print("CONFIDENCE:", confidence)
    print("CONFIDENCE %:", confidence * 100)
    print("========================================")

    if confidence < 0.40:
        return jsonify({
            "status": "low_confidence",
            "message": "The model is not confident enough about this image. Please upload a clear close-up photo of the leaf.",
            "confidence": round(confidence * 100, 1),
        })

    raw_label = LABELS[best_index]
    parts = raw_label.split("___")
    predicted_crop_raw = parts[0]
    predicted_disease = parts[1] if len(parts) > 1 else raw_label
    detected_crop_id = MODEL_CROP_TO_APP_CROP_ID.get(predicted_crop_raw)

    # Guard 2: model detected a different crop than the one selected.
    if detected_crop_id != crop_id:
        return jsonify({
            "status": "crop_mismatch",
            "message": "Image does not appear to match the selected crop.",
            "selected_crop": crop_id,
            "detected_crop": detected_crop_id,
        })

    is_healthy = predicted_disease.lower() == "healthy"
    tier = confidence_tier(confidence)

    if is_healthy:
        result = {
            "status": "success",
            "crop": detected_crop_id,
            "disease": "Healthy",
            "is_healthy": True,
            "confidence": round(confidence * 100, 1),
            "confidence_tier": tier,
            "severity": "none",
            "symptoms": ["No visible signs of disease were detected in the image."],
            "recommendations": ["Continue routine monitoring and good field/orchard practices."],
            "prevention": ["Keep monitoring regularly, since new symptoms can appear over time."],
        }
    else:
        info = KNOWLEDGE_BASE.get((predicted_crop_raw, predicted_disease))
        if info is None:
            info = dict(
                symptoms=["Detailed information is not yet available in the local knowledge base."],
                recommendations=["Consult a local agricultural expert before applying any treatment."],
                prevention=["Use verified crop-management guidance for this disease."],
            )
        result = {
            "status": "success",
            "crop": detected_crop_id,
            "disease": predicted_disease.replace("_", " "),
            "is_healthy": False,
            "confidence": round(confidence * 100, 1),
            "confidence_tier": tier,
            "severity": severity_from_tier(tier, False),
            "symptoms": info["symptoms"],
            "recommendations": info["recommendations"],
            "prevention": info["prevention"],
        }

    if tier == "low":
        result["warning"] = "Confidence is low — consider retaking the photo in better light, closer up."

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    conn = get_db()
    if conn is None:
        return jsonify(list(reversed(_memory_history)))
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM analysis_history ORDER BY created_at DESC LIMIT 200;")
        rows = cur.fetchall()
    conn.close()
    for r in rows:
        r["id"] = str(r["id"])
        r["created_at"] = r["created_at"].isoformat()
    return jsonify(rows)


@app.route("/api/history", methods=["POST"])
def save_history():
    data = request.get_json(force=True, silent=True) or {}
    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "crop": data.get("crop", "unknown"),
        "disease": data.get("disease", "unknown"),
        "confidence": float(data.get("confidence", 0)),
        "severity": data.get("severity", "none"),
        "status": data.get("status", "success"),
    }
    conn = get_db()
    if conn is None:
        _memory_history.append(record)
        return jsonify(record), 201
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO analysis_history (id, crop, disease, confidence, severity, status) "
            "VALUES (%s, %s, %s, %s, %s, %s);",
            (record["id"], record["crop"], record["disease"], record["confidence"],
             record["severity"], record["status"]),
        )
    conn.close()
    return jsonify(record), 201


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_history_item(record_id):
    conn = get_db()
    if conn is None:
        global _memory_history
        _memory_history = [r for r in _memory_history if r["id"] != record_id]
        return jsonify({"deleted": record_id})
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM analysis_history WHERE id = %s;", (record_id,))
    conn.close()
    return jsonify({"deleted": record_id})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    conn = get_db()
    if conn is None:
        _memory_history.clear()
        return jsonify({"cleared": True})
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM analysis_history;")
    conn.close()
    return jsonify({"cleared": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Farmer-friendly AI Assistant, backed by Groq's free LLM API.

    The ML diagnosis (if any) is passed in as context and the model is
    explicitly told it is the source of truth — the LLM explains it, it
    never invents or overrides a diagnosis.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    diagnosis = data.get("diagnosis")  # optional dict from a prior /api/analyze call

    if not question:
        return jsonify({"error": "No question provided."}), 400

    if not GROQ_API_KEY:
        return jsonify({
            "answer": "The AI Assistant isn't connected yet. Add a free GROQ_API_KEY "
                      "(from console.groq.com) to the backend's environment variables on Render.",
            "connected": False,
        })

    if diagnosis:
        context = (
            f"Crop: {diagnosis.get('crop')}\n"
            f"Disease: {diagnosis.get('disease')}\n"
            f"Confidence: {diagnosis.get('confidence')}%\n"
            f"Severity: {diagnosis.get('severity')}\n"
            f"Symptoms: {', '.join(diagnosis.get('symptoms', []))}\n"
            f"Recommendations: {', '.join(diagnosis.get('recommendations', []))}\n"
            f"Prevention: {', '.join(diagnosis.get('prevention', []))}"
        )
    else:
        context = "No diagnosis has been run yet."

    system_prompt = (
        "You are Crop Doctor's AI Assistant, helping farmers understand a plant disease "
        "diagnosis in simple, friendly, non-technical language. "
        "The diagnosis below came from a real trained ML model and is the ONLY source of "
        "truth for what disease was detected — never contradict it, never invent a "
        "different crop or disease. You only explain, advise, and answer follow-up "
        "questions about it. Keep answers short (2-5 sentences) and practical.\n\n"
        f"Current diagnosis:\n{context}"
    )

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "temperature": 0.4,
                "max_tokens": 300,
            },
            timeout=20,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"answer": answer, "connected": True})
    except Exception as e:
        print("GROQ ERROR:", repr(e), flush=True)

        return jsonify({
            "answer": "The AI Assistant is temporarily unavailable. Please try again in a moment.",
            "connected": False,
        }), 503

load_model()
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

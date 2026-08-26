"""
Crop Doctor — Backend API
==========================
Flask API serving the Crop Doctor 39-class MobileNetV2 model
(CropDoctor_39Class_MobileNetV2.tflite) with structured, per-class
diagnosis information, and analysis history in a Neon (serverless
Postgres) database.

Deploy target: Render (Web Service)
Database:      Neon Postgres (set DATABASE_URL env var)
Frontend:      Netlify (calls this API, set FRONTEND_ORIGIN env var for CORS)

Endpoints
---------
GET  /api/health              -> {"status": "ok", "model_loaded": true/false}
POST /api/analyze              -> multipart/form-data: image=<file>
                                   returns diagnosis JSON (never fakes a result)
GET  /api/classes              -> list of the 39 supported classes
GET  /api/history              -> list of saved analyses (most recent first)
POST /api/history              -> save an analysis record
DELETE /api/history/<id>       -> delete one record
DELETE /api/history            -> clear all history

Nothing here invents an ML prediction. If the model file is missing or
fails to load, /api/analyze returns a clear "model unavailable" error
instead of a fabricated diagnosis.
"""

import io
import os
import json
import uuid
import datetime
import logging

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf

Interpreter = tf.lite.Interpreter

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CropDoctor_39Class_MobileNetV2.tflite")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

# Groq (cloud LLM) settings for the optional AI Assistant chat feature.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Model input size / preprocessing. The model was trained on 224x224 RGB,
# float32 input. Pixel scaling assumption: 0..1 (i.e. raw_pixel / 255.0),
# matching the standard MobileNetV2-transfer-learning recipe used for this
# project. If validation accuracy looks noticeably worse in production than
# the ~84.9% reported during training, the most likely cause is a mismatch
# here — try IMG_MEAN=127.5, IMG_STD=127.5 (i.e. [-1, 1] scaling) instead.
IMG_SIZE = (224, 224)
IMG_MEAN = 0.0
IMG_STD = 255.0

# Confidence thresholds used to decide how strongly to present a result.
# Documented here since they directly affect what the user is told.
CONFIDENCE_HIGH = 0.75      # >=75%  -> present result normally
CONFIDENCE_MODERATE = 0.45  # >=45%  -> present result, advise verification
# < 45% -> explicitly flagged as uncertain

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONTEND_ORIGIN}})
logging.basicConfig(level=logging.INFO)
log = app.logger

# ---------------------------------------------------------------------------
# Class names — loaded from class_names.json (source of truth for label
# order). Never hardcode a different order than what's in that file.
# ---------------------------------------------------------------------------
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)  # expected: 39


def _split_crop_condition(class_name: str):
    """Splits e.g. 'Bell_Pepper_Bacterial_Spot' into ('Bell Pepper',
    'Bacterial Spot') using the known crop-name prefixes so multi-word
    crops (Bell Pepper) are handled correctly."""
    known_crops = ["Bell_Pepper", "Apple", "Corn", "Grape", "Potato",
                   "Rice", "Tomato", "Wheat"]
    for crop in known_crops:
        if class_name.startswith(crop + "_"):
            crop_display = crop.replace("_", " ")
            condition_raw = class_name[len(crop) + 1:]
            return crop_display, condition_raw
    # Fallback: split on first underscore
    parts = class_name.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _display_condition(condition_raw: str) -> str:
    return condition_raw.replace("_", " ")


# ---------------------------------------------------------------------------
# Curated agricultural knowledge base — one entry per class in
# class_names.json. General, responsible management guidance; not a
# substitute for a local agricultural expert.
# ---------------------------------------------------------------------------
CLASS_INFO = {
    "Apple_Apple_Scab": dict(
        cause="The fungus Venturia inaequalis, which thrives in cool, wet spring weather and overwinters in fallen leaves.",
        symptoms=["Olive-green to dark brown velvety spots on leaves",
                   "Corky, scab-like lesions on fruit skin",
                   "Premature yellowing and leaf drop"],
        action=["Remove and destroy fallen leaves to cut down overwintering spores",
                "Apply a labeled fungicide starting at green-tip and repeat per the product interval",
                "Prune to open up the canopy and improve air circulation"],
        prevention=["Choose scab-resistant apple varieties where possible",
                     "Avoid overhead irrigation that keeps foliage wet",
                     "Rake and dispose of leaf litter every autumn"],
        severity="Attention recommended",
    ),
    "Apple_Black_Rot": dict(
        cause="The fungus Botryosphaeria obtusa, entering through wounds, dead wood, or old fruit mummies.",
        symptoms=["Purple-bordered \"frog-eye\" spots on leaves",
                   "Rotting, mummified fruit with concentric rings",
                   "Sunken, reddish-brown cankers on branches"],
        action=["Prune out and destroy cankered wood and mummified fruit",
                "Apply a labeled fungicide from bloom through early summer",
                "Remove nearby dead or diseased wood that can harbor spores"],
        prevention=["Sanitize pruning tools between cuts",
                     "Avoid wounding bark and fruit during harvest",
                     "Maintain tree vigor with balanced fertilization"],
        severity="Attention recommended",
    ),
    "Apple_Cedar_Apple_Rust": dict(
        cause="The fungus Gymnosporangium juniperi-virginianae, which requires nearby juniper/cedar hosts to complete its life cycle.",
        symptoms=["Bright yellow-orange spots on the upper leaf surface",
                   "Small raised cups or tubes on the leaf underside",
                   "Distorted or spotted fruit in severe cases"],
        action=["Apply a protective fungicide from pink bud through early summer",
                "Remove nearby juniper/cedar hosts within a few hundred meters if feasible",
                "Rake and destroy fallen infected leaves"],
        prevention=["Plant rust-resistant apple varieties",
                     "Avoid planting apples near ornamental junipers",
                     "Monitor closely during wet spring weather"],
        severity="Attention recommended",
    ),
    "Apple_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Bell_Pepper_Bacterial_Spot": dict(
        cause="Xanthomonas bacteria, spread by splashing water, contaminated seed, and handling wet plants.",
        symptoms=["Small, water-soaked spots on leaves that turn brown and scab-like",
                   "Raised, rough spots on the fruit surface",
                   "Leaf yellowing and drop in severe cases"],
        action=["Remove and destroy severely infected plants/leaves",
                "Apply a copper-based bactericide early, per local label guidance",
                "Avoid working among wet plants to limit spread"],
        prevention=["Use certified disease-free seed and transplants",
                     "Rotate away from peppers/tomatoes for 1–2 seasons",
                     "Water at the base of plants, not overhead"],
        severity="Attention recommended",
    ),
    "Bell_Pepper_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Corn_Common_Rust": dict(
        cause="The fungus Puccinia sorghi, favored by cool, humid weather.",
        symptoms=["Small, cinnamon-brown, powdery pustules on both leaf surfaces",
                   "Pustules rupture the leaf surface, releasing rust-colored spores",
                   "Heavily infected leaves may yellow and die prematurely"],
        action=["Apply a fungicide if pustules appear early and conditions stay cool/humid",
                "Favor rust-resistant hybrids in future seasons",
                "Scout fields regularly during cool, humid weather"],
        prevention=["Plant resistant hybrids where common rust recurs",
                     "Avoid excessive nitrogen that promotes dense, humid canopies",
                     "Scout early since rust can spread quickly"],
        severity="Attention recommended",
    ),
    "Corn_Gray_Leaf_Spot": dict(
        cause="The fungus Cercospora zeae-maydis, which overwinters in corn residue and favors humid, warm conditions.",
        symptoms=["Small, rectangular tan-to-gray lesions bound by leaf veins",
                   "Lesions merge in severe infections, blighting whole leaves",
                   "Symptoms usually start on lower leaves and move upward"],
        action=["Apply a foliar fungicide if disease appears before tasseling",
                "Rotate with a non-host crop for at least one season",
                "Select resistant hybrids for future plantings"],
        prevention=["Manage crop residue/tillage to reduce overwintering spores",
                     "Avoid continuous corn-on-corn planting in the same field",
                     "Ensure adequate plant spacing for airflow"],
        severity="Attention recommended",
    ),
    "Corn_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Corn_Northern_Leaf_Blight": dict(
        cause="The fungus Exserohilum turcicum, favored by moderate temperatures and extended leaf wetness.",
        symptoms=["Long, cigar-shaped gray-green to tan lesions on leaves",
                   "Lesions can span several centimeters and merge together",
                   "Severe infection causes premature leaf death and yield loss"],
        action=["Apply a labeled fungicide, especially at or before tasseling",
                "Rotate crops away from corn for at least one year",
                "Till under crop residue where the fungus overwinters"],
        prevention=["Plant hybrids with genetic resistance to Northern Leaf Blight",
                     "Avoid dense planting that keeps the canopy humid",
                     "Scout fields regularly after warm, wet weather"],
        severity="Attention recommended",
    ),
    "Grape_Black_Rot": dict(
        cause="The fungus Guignardia bidwellii, which overwinters in mummified berries and infected canes.",
        symptoms=["Small tan spots with dark borders on leaves",
                   "Fruit shrivels into hard, black \"mummies\"",
                   "Reddish-brown lesions can appear on shoots and tendrils"],
        action=["Remove and destroy mummified berries and infected leaves/canes",
                "Apply a fungicide from early shoot growth through veraison per label",
                "Improve canopy airflow through timely pruning"],
        prevention=["Prune out infected wood during dormancy",
                     "Avoid working in the vineyard when foliage is wet",
                     "Choose sites with good air movement and sun exposure"],
        severity="Attention recommended",
    ),
    "Grape_Esca": dict(
        cause="A complex of wood-rotting fungi that enter through pruning wounds; chronic and hard to eliminate once established.",
        symptoms=["\"Tiger-stripe\" interveinal yellowing/browning on leaves",
                   "Dark spotting on berries in severe cases",
                   "Internal wood streaking and dieback of cordons/trunk"],
        action=["Remove and destroy severely affected vines/wood to slow spread",
                "Protect large pruning wounds with a wound sealant",
                "Avoid pruning during wet weather, when infection risk is highest"],
        prevention=["Use delayed or double pruning to reduce wound exposure time",
                     "Sanitize pruning tools between vines",
                     "Maintain overall vine health to improve tolerance"],
        severity="High attention",
    ),
    "Grape_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Grape_Leaf_Blight": dict(
        cause="The fungus Pseudocercospora vitis (Isariopsis leaf spot), favored by warm, humid conditions.",
        symptoms=["Angular brown-to-black spots on leaves, often with a yellow halo",
                   "Spots may merge, leading to premature leaf drop",
                   "Reduced vine vigor with repeated defoliation"],
        action=["Apply a protective fungicide during the growing season per label",
                "Remove fallen, infected leaves at season end",
                "Improve canopy ventilation through leaf pulling/pruning"],
        prevention=["Avoid overhead irrigation that prolongs leaf wetness",
                     "Maintain good weed control to improve airflow near the ground",
                     "Monitor closely during warm, humid periods"],
        severity="Attention recommended",
    ),
    "Potato_Early_Blight": dict(
        cause="The fungus Alternaria solani, favored by warm temperatures and periods of leaf wetness; often first on stressed or older foliage.",
        symptoms=["Dark, concentric \"target-ring\" spots on older leaves first",
                   "Yellowing tissue surrounding leaf spots",
                   "Lesions can also appear on stems and tubers"],
        action=["Remove and destroy heavily infected lower leaves",
                "Apply a labeled fungicide on a preventive schedule in humid weather",
                "Maintain balanced fertility — stressed plants are more susceptible"],
        prevention=["Rotate potatoes with non-host crops for 2+ years",
                     "Space plants for good airflow and faster leaf drying",
                     "Avoid overhead irrigation late in the day"],
        severity="Attention recommended",
    ),
    "Potato_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Potato_Late_Blight": dict(
        cause="The oomycete Phytophthora infestans — spreads very fast in cool, wet weather and can destroy a field within days.",
        symptoms=["Water-soaked, pale-to-dark green lesions that turn brown/black quickly",
                   "White fungal growth on the leaf underside in humid conditions",
                   "Firm, dark, granular rot on tubers"],
        action=["Act quickly: remove and destroy infected foliage/plants",
                "Apply a labeled fungicide immediately — this disease spreads fast",
                "Avoid irrigating or harvesting in wet conditions once detected"],
        prevention=["Plant certified, disease-free seed potatoes",
                     "Destroy volunteer potato plants and cull piles",
                     "Monitor closely during cool, wet weather"],
        severity="High attention",
    ),
    "Rice_Bacterial_Leaf_Blight": dict(
        cause="Xanthomonas oryzae bacteria, spreading through irrigation water, wind-driven rain, and wounds.",
        symptoms=["Water-soaked streaks near leaf tips/margins that turn yellow to white",
                   "Wavy lesion borders that expand along the leaf",
                   "Wilting of young seedlings (\"kresek\") in severe cases"],
        action=["Drain standing water where possible to reduce bacterial spread",
                "Avoid excess nitrogen fertilizer, which increases susceptibility",
                "Apply a locally recommended bactericide/copper product if severe"],
        prevention=["Use certified, disease-free seed and resistant varieties",
                     "Avoid field-to-field water flow from infected fields",
                     "Practice balanced fertilization"],
        severity="Attention recommended",
    ),
    "Rice_Brown_Spot": dict(
        cause="The fungus Bipolaris oryzae, often linked to nutrient-poor or stressed soils.",
        symptoms=["Small, oval brown spots with gray-white centers on leaves",
                   "Spots can also appear on the grain, causing discoloration",
                   "More common in nutrient-deficient fields"],
        action=["Apply a labeled fungicide if the outbreak is severe near heading",
                "Improve field nutrition, particularly potassium",
                "Use balanced fertilization to strengthen plant resistance"],
        prevention=["Use certified, healthy seed",
                     "Maintain adequate soil fertility and drainage",
                     "Avoid water stress during the growing season"],
        severity="Monitor",
    ),
    "Rice_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Rice_Leaf_Blast": dict(
        cause="The fungus Magnaporthe oryzae — one of the most destructive rice diseases worldwide, favored by high humidity and dense canopies.",
        symptoms=["Diamond/spindle-shaped lesions with gray centers and brown borders",
                   "Lesions can merge and kill entire leaves",
                   "Neck/panicle infection can cause severe yield loss"],
        action=["Apply a labeled fungicide promptly, especially before heading",
                "Avoid excess nitrogen, which increases susceptibility",
                "Improve field drainage and avoid dense planting"],
        prevention=["Plant blast-resistant varieties where available",
                     "Use balanced nitrogen application, split across the season",
                     "Monitor closely during humid, cloudy weather"],
        severity="High attention",
    ),
    "Rice_Leaf_Scald": dict(
        cause="The fungus Microdochium oryzae, often associated with dense canopies and excess nitrogen.",
        symptoms=["Zonate, alternating light-and-dark bands on leaf tips/margins",
                   "Scalded, straw-colored appearance on affected tissue",
                   "Lesions expand from leaf tips inward"],
        action=["Improve field drainage and airflow through the canopy",
                "Avoid excess nitrogen fertilization",
                "Apply a labeled fungicide if the outbreak is severe"],
        prevention=["Use resistant varieties where available",
                     "Balance nitrogen application across the season",
                     "Avoid dense planting"],
        severity="Monitor",
    ),
    "Rice_Sheath_Blight": dict(
        cause="The fungus Rhizoctonia solani, thriving in dense, humid canopies with high nitrogen.",
        symptoms=["Oval, greenish-gray lesions with irregular borders on leaf sheaths",
                   "Lesions can climb the plant and merge together",
                   "Can cause lodging and reduced grain fill in severe cases"],
        action=["Apply a labeled fungicide at early symptom onset",
                "Reduce planting density to improve airflow",
                "Avoid excess nitrogen fertilization"],
        prevention=["Maintain proper plant spacing",
                     "Use balanced fertilization, avoiding nitrogen excess",
                     "Improve field drainage between irrigations"],
        severity="Attention recommended",
    ),
    "Tomato_Bacterial_Spot": dict(
        cause="Xanthomonas bacteria, spread by splashing water, contaminated seed, and handling wet plants.",
        symptoms=["Small, water-soaked, greasy-looking leaf spots that turn dark",
                   "Raised, scabby spots on green fruit",
                   "Leaf yellowing and defoliation in severe cases"],
        action=["Remove and destroy severely infected leaves/plants",
                "Apply a copper-based bactericide per local label guidance",
                "Avoid handling wet plants to limit spread between them"],
        prevention=["Use certified disease-free seed and transplants",
                     "Rotate away from tomatoes/peppers for 1–2 seasons",
                     "Water at the base of plants, not overhead"],
        severity="Attention recommended",
    ),
    "Tomato_Early_Blight": dict(
        cause="The fungus Alternaria solani, favored by warm temperatures and periods of leaf wetness.",
        symptoms=["Dark, concentric \"target-ring\" spots on older/lower leaves",
                   "Yellow halo surrounding leaf spots",
                   "Dark, leathery lesions can form near the fruit stem"],
        action=["Remove and destroy infected lower leaves promptly",
                "Apply a labeled fungicide on a preventive schedule in humid weather",
                "Stake or cage plants to keep foliage off the soil"],
        prevention=["Rotate tomatoes with non-host crops for 2+ years",
                     "Mulch to reduce soil splashing spores onto leaves",
                     "Avoid overhead irrigation late in the day"],
        severity="Attention recommended",
    ),
    "Tomato_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Tomato_Late_Blight": dict(
        cause="The oomycete Phytophthora infestans — spreads very fast in cool, wet weather.",
        symptoms=["Large, water-soaked, pale-green to brown blotches on leaves",
                   "White, fuzzy fungal growth on leaf undersides in humid weather",
                   "Firm, dark, greasy-looking rot on fruit"],
        action=["Act quickly: remove and destroy infected foliage/plants",
                "Apply a labeled fungicide immediately — this disease spreads fast",
                "Avoid overhead watering once detected"],
        prevention=["Space and stake plants for good airflow",
                     "Avoid planting near infected potato fields",
                     "Monitor closely during cool, wet, humid weather"],
        severity="High attention",
    ),
    "Tomato_Leaf_Mold": dict(
        cause="The fungus Passalora fulva (syn. Fulvia fulva), thriving in humid, poorly ventilated conditions, especially greenhouses.",
        symptoms=["Pale green-to-yellow spots on the upper leaf surface",
                   "Olive-green to grayish-brown fuzzy mold on the underside",
                   "Common in humid greenhouse/covered growing conditions"],
        action=["Improve ventilation and reduce humidity around plants",
                "Remove and destroy heavily infected leaves",
                "Apply a labeled fungicide if conditions stay humid"],
        prevention=["Space plants and prune to improve air circulation",
                     "Water at the base of plants, avoiding wet foliage",
                     "Choose resistant tomato varieties where available"],
        severity="Attention recommended",
    ),
    "Tomato_Mosaic_Virus": dict(
        cause="Tomato mosaic virus (ToMV), easily spread by hand contact, tools, and infected seed. There is no cure once a plant is infected.",
        symptoms=["Light and dark green mottled/mosaic pattern on leaves",
                   "Leaf distortion, curling, or fern-like narrowing",
                   "Stunted growth and reduced, sometimes mottled, fruit"],
        action=["Remove and destroy infected plants — there is no cure",
                "Wash hands and disinfect tools after handling infected plants",
                "Avoid tobacco use near plants; the virus can be transmitted this way"],
        prevention=["Use certified virus-free seed and resistant varieties",
                     "Disinfect tools and hands between plants when pruning/staking",
                     "Control weeds that may harbor the virus"],
        severity="High attention",
    ),
    "Tomato_Septoria_Leaf_Spot": dict(
        cause="The fungus Septoria lycopersici, spread by splashing water and favored by humid conditions.",
        symptoms=["Numerous small, circular spots with dark borders and gray centers",
                   "Tiny black specks (fungal fruiting bodies) visible in spot centers",
                   "Lower leaves affected first, often causing yellowing and drop"],
        action=["Remove and destroy infected lower leaves promptly",
                "Apply a labeled fungicide on a preventive schedule",
                "Avoid working among wet plants to limit spread"],
        prevention=["Rotate tomatoes with non-host crops for at least one season",
                     "Mulch to reduce soil splash onto lower leaves",
                     "Stake or cage plants to keep foliage off the ground"],
        severity="Attention recommended",
    ),
    "Tomato_Spider_Mites": dict(
        cause="Two-spotted spider mites (Tetranychus urticae), tiny pests that thrive in hot, dry conditions.",
        symptoms=["Fine yellow/white stippling on leaf surfaces",
                   "Fine webbing on leaves and stems in heavy infestations",
                   "Leaves may bronze, dry out, and drop in severe cases"],
        action=["Rinse plants with a strong water spray to dislodge mites",
                "Apply an appropriate miticide or insecticidal soap if severe",
                "Introduce or conserve natural predators where feasible"],
        prevention=["Avoid drought-stressing plants, which favors mite outbreaks",
                     "Monitor closely during hot, dry weather",
                     "Remove heavily infested leaves early"],
        severity="Attention recommended",
    ),
    "Tomato_Target_Spot": dict(
        cause="The fungus Corynespora cassiicola, favored by warm, humid weather.",
        symptoms=["Brown, concentric-ringed spots on leaves, stems and fruit",
                   "Spots may merge, causing large blighted areas",
                   "Can lead to significant defoliation in humid conditions"],
        action=["Remove and destroy infected leaves/debris",
                "Apply a labeled fungicide on a preventive schedule in humid weather",
                "Improve airflow through staking and pruning"],
        prevention=["Rotate with non-host crops where practical",
                     "Avoid overhead irrigation late in the day",
                     "Space plants for good ventilation"],
        severity="Attention recommended",
    ),
    "Tomato_Yellow_Leaf_Curl_Virus": dict(
        cause="Tomato yellow leaf curl virus (TYLCV), transmitted by whiteflies. There is no cure once a plant is infected.",
        symptoms=["Upward curling and yellowing of leaf edges",
                   "Stunted, bushy growth habit",
                   "Significant reduction in flowering and fruit set"],
        action=["Remove and destroy infected plants to reduce virus spread",
                "Control whitefly populations with appropriate measures",
                "Use reflective mulches or netting to deter whiteflies where practical"],
        prevention=["Use virus-resistant tomato varieties where available",
                     "Manage whiteflies proactively, especially early in the season",
                     "Avoid planting new tomatoes next to infected fields"],
        severity="High attention",
    ),
    "Wheat_Healthy": dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Healthy",
    ),
    "Wheat_Leaf_Rust": dict(
        cause="The fungus Puccinia triticina, favored by moderate temperatures and moisture.",
        symptoms=["Small, round, orange-brown pustules scattered on leaf surfaces",
                   "Pustules rupture the leaf surface, releasing rust-colored spores",
                   "Premature leaf senescence in severe infections"],
        action=["Apply a labeled fungicide if detected early in the season",
                "Favor rust-resistant varieties for future planting",
                "Scout fields regularly during moderate, humid weather"],
        prevention=["Plant resistant wheat varieties where available",
                     "Avoid excess nitrogen that promotes dense, humid canopies",
                     "Monitor regional rust forecasts/advisories"],
        severity="Attention recommended",
    ),
    "Wheat_Powdery_Mildew": dict(
        cause="The fungus Blumeria graminis, favored by cool temperatures, high humidity, and dense canopies.",
        symptoms=["White to grayish powdery patches on leaves and stems",
                   "Patches may turn yellow-brown as the disease progresses",
                   "Reduced vigor and yield in heavily infected fields"],
        action=["Apply a labeled fungicide if detected early, especially in dense stands",
                "Improve airflow through appropriate seeding rates",
                "Avoid excess nitrogen fertilization"],
        prevention=["Plant resistant varieties where available",
                     "Avoid overly dense planting",
                     "Balance nitrogen application across the season"],
        severity="Monitor",
    ),
    "Wheat_Septoria": dict(
        cause="The fungus Zymoseptoria tritici (Septoria leaf blotch), spread by rain splash and favored by cool, wet weather.",
        symptoms=["Irregular tan-to-brown blotches on leaves with tiny black fruiting bodies",
                   "Lesions often start on lower leaves and move upward",
                   "Can significantly reduce green leaf area during grain fill"],
        action=["Apply a labeled fungicide at early symptom onset, per local guidance",
                "Rotate away from wheat/cereal crops where practical",
                "Manage crop residue to reduce carryover inoculum"],
        prevention=["Plant resistant or tolerant varieties where available",
                     "Avoid excessive seeding rates that increase canopy humidity",
                     "Monitor closely during cool, wet stretches"],
        severity="Attention recommended",
    ),
    "Wheat_Stem_Rust": dict(
        cause="The fungus Puccinia graminis f. sp. tritici — historically one of the most destructive wheat diseases, favored by warm, humid conditions.",
        symptoms=["Large, dark reddish-brown, raised pustules mainly on stems and leaf sheaths",
                   "Pustules rupture the surface, exposing powdery rust-colored spores",
                   "Can cause severe stem breakage (lodging) and yield loss"],
        action=["Apply a labeled fungicide promptly — stem rust can spread quickly",
                "Favor resistant varieties for future planting",
                "Report suspected outbreaks to local agricultural authorities where relevant"],
        prevention=["Plant resistant wheat varieties where available",
                     "Monitor regional rust forecasts/advisories closely",
                     "Avoid late-season nitrogen that prolongs canopy humidity"],
        severity="High attention",
    ),
    "Wheat_Yellow_Rust": dict(
        cause="The fungus Puccinia striiformis (stripe rust), favored by cool, humid conditions and capable of spreading over long distances.",
        symptoms=["Yellow-orange pustules arranged in narrow stripes along leaf veins",
                   "Stripes most visible on upper leaf surfaces",
                   "Can cause significant yield loss if it strikes early in the season"],
        action=["Apply a labeled fungicide promptly if detected — stripe rust spreads fast in cool weather",
                "Favor resistant varieties for future planting",
                "Scout fields regularly during cool, humid periods"],
        prevention=["Plant resistant wheat varieties where available",
                     "Monitor regional rust forecasts/advisories closely",
                     "Avoid excess nitrogen that promotes dense, humid canopies"],
        severity="Attention recommended",
    ),
}

# Sanity check at import time — every class in class_names.json must have
# a knowledge-base entry, and vice versa.
_missing = [c for c in CLASS_NAMES if c not in CLASS_INFO]
if _missing:
    log.warning("CLASS_INFO is missing entries for: %s", _missing)

# ---------------------------------------------------------------------------
# Model loading (once, at process start; reused for every request)
# ---------------------------------------------------------------------------
_interpreter = None
_input_details = None
_output_details = None


def load_model():
    global _interpreter, _input_details, _output_details
    if _interpreter is not None:
        return
    if not os.path.exists(MODEL_PATH):
        log.error("Model file not found at %s", MODEL_PATH)
        return
    try:
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
        log.info("Model loaded: %s classes expected, got %s output classes",
                  NUM_CLASSES, _output_details[0]["shape"][-1])
    except Exception:
        log.exception("Failed to load TFLite model")
        _interpreter = None


def confidence_tier(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MODERATE:
        return "moderate"
    return "low"


def run_inference(image_bytes: bytes):
    """Runs the 39-class MobileNetV2 tflite model on the given image bytes.
    Returns the raw probability array of length NUM_CLASSES."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = np.expand_dims(arr, axis=0)

    _interpreter.set_tensor(_input_details[0]["index"], arr)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]["index"])
    return output[0]


# ---------------------------------------------------------------------------
# Database (Neon Postgres) — analysis history
# ---------------------------------------------------------------------------
_memory_history = []  # fallback store used only when DATABASE_URL isn't set


def get_db():
    if not DATABASE_URL or psycopg2 is None:
        return None
    try:
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    except Exception:
        log.exception("Database connection failed")
        return None


def init_db():
    conn = get_db()
    if conn is None:
        return
    try:
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
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _interpreter is not None,
        "num_classes": NUM_CLASSES,
        "database_connected": DATABASE_URL is not None,
    })


@app.route("/api/classes", methods=["GET"])
def classes():
    return jsonify(CLASS_NAMES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"status": "error", "message": "No image was uploaded."}), 400

    if _interpreter is None:
        load_model()
    if _interpreter is None:
        return jsonify({
            "status": "model_unavailable",
            "message": "The diagnosis model is temporarily unavailable. Please try again shortly.",
        }), 503

    try:
        image_bytes = image_file.read()
        if not image_bytes:
            raise ValueError("empty file")
        probabilities = run_inference(image_bytes)
    except UnidentifiedImageError:
        return jsonify({
            "status": "image_unreadable",
            "message": "That file doesn't look like a valid image. Please upload a JPG or PNG photo.",
        }), 400
    except Exception:
        log.exception("Inference failed")
        return jsonify({
            "status": "image_unreadable",
            "message": "That image couldn't be processed. Please try a clearer photo.",
        }), 400

    if len(probabilities) != NUM_CLASSES:
        log.error("Model output size %s does not match %s classes", len(probabilities), NUM_CLASSES)
        return jsonify({
            "status": "model_unavailable",
            "message": "The diagnosis model returned an unexpected result. Please try again shortly.",
        }), 503

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    class_name = CLASS_NAMES[best_index]
    crop_display, condition_raw = _split_crop_condition(class_name)
    condition_display = _display_condition(condition_raw)
    is_healthy = condition_raw.lower() == "healthy"
    tier = confidence_tier(confidence)

    info = CLASS_INFO.get(class_name, dict(
        cause="", symptoms=[], action=[], prevention=[], severity="Monitor",
    ))

    if is_healthy:
        result = {
            "status": "success",
            "crop": crop_display,
            "condition": "Healthy",
            "is_healthy": True,
            "confidence": round(confidence * 100, 1),
            "confidence_tier": tier,
            "severity": "Healthy",
            "explanation": f"No signs of disease were detected on this {crop_display.lower()} leaf.",
            "cause": "",
            "symptoms": ["No visible disease symptoms were detected in the image."],
            "action": ["Continue routine monitoring and good field/orchard practices."],
            "prevention": ["Keep monitoring regularly — new symptoms can appear over time."],
        }
    else:
        result = {
            "status": "success",
            "crop": crop_display,
            "condition": condition_display,
            "is_healthy": False,
            "confidence": round(confidence * 100, 1),
            "confidence_tier": tier,
            "severity": info["severity"],
            "explanation": f"The image shows signs consistent with {condition_display.lower()} on {crop_display.lower()}.",
            "cause": info["cause"],
            "symptoms": info["symptoms"],
            "action": info["action"],
            "prevention": info["prevention"],
        }

    if tier == "moderate":
        result["warning"] = "Moderate confidence — this result is likely but worth confirming, especially before taking action."
    elif tier == "low":
        result["warning"] = "Low confidence — this prediction is uncertain. Try a clearer, closer photo in good light, or have a local expert confirm."

    result["disclaimer"] = ("This is an AI-assisted assessment, not a guaranteed laboratory diagnosis. "
                             "For high-value crops or uncertain cases, confirm with a local agricultural expert.")

    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({
            "status": "error",
            "message": "AI Assistant is not configured (missing GROQ_API_KEY on the server).",
        }), 503

    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    context = data.get("context")  # optional diagnosis context dict from the frontend

    if not question:
        return jsonify({"status": "error", "message": "No question was provided."}), 400

    if context:
        context_text = (
            f"Crop: {context.get('crop', 'unknown')}\n"
            f"Condition: {context.get('condition', 'unknown')}\n"
            f"Confidence: {context.get('confidence', 'unknown')}%\n"
            f"Severity: {context.get('severity', 'unknown')}\n"
            f"Symptoms: {'; '.join(context.get('symptoms', []))}\n"
            f"Recommended action: {'; '.join(context.get('action', []))}"
        )
    else:
        context_text = "No diagnosis has been run yet."

    system_prompt = (
        "You are a farmer-friendly crop health assistant. Explain things in simple, "
        "practical language. The ML model's diagnosis is the ONLY source of truth for "
        "what was detected — never invent or contradict a different diagnosis. "
        "Keep answers concise (a few sentences to a short paragraph)."
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Diagnosis context:\n{context_text}\n\nFarmer's question: {question}"},
                ],
                "temperature": 0.4,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        answer = payload["choices"][0]["message"]["content"].strip()
        return jsonify({"status": "success", "answer": answer})
    except requests.exceptions.RequestException:
        log.exception("GROQ request failed")
        return jsonify({
            "status": "error",
            "message": "The AI Assistant is temporarily unavailable. Please try again in a moment.",
        }), 503


@app.route("/api/history", methods=["GET"])
def get_history():
    conn = get_db()
    if conn is None:
        return jsonify(list(reversed(_memory_history)))
    try:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM analysis_history ORDER BY created_at DESC LIMIT 200;")
            rows = cur.fetchall()
        for r in rows:
            r["id"] = str(r["id"])
            r["created_at"] = r["created_at"].isoformat()
        return jsonify(rows)
    finally:
        conn.close()


@app.route("/api/history", methods=["POST"])
def save_history():
    data = request.get_json(force=True, silent=True) or {}
    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "crop": data.get("crop", "unknown"),
        "disease": data.get("condition", data.get("disease", "unknown")),
        "confidence": float(data.get("confidence", 0)),
        "severity": data.get("severity", "none"),
        "status": data.get("status", "success"),
    }
    conn = get_db()
    if conn is None:
        _memory_history.append(record)
        return jsonify(record), 201
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO analysis_history (id, crop, disease, confidence, severity, status) "
                "VALUES (%s, %s, %s, %s, %s, %s);",
                (record["id"], record["crop"], record["disease"], record["confidence"],
                 record["severity"], record["status"]),
            )
        return jsonify(record), 201
    finally:
        conn.close()


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_history_item(record_id):
    conn = get_db()
    if conn is None:
        global _memory_history
        _memory_history = [r for r in _memory_history if r["id"] != record_id]
        return jsonify({"deleted": record_id})
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM analysis_history WHERE id = %s;", (record_id,))
        return jsonify({"deleted": record_id})
    finally:
        conn.close()


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    conn = get_db()
    if conn is None:
        _memory_history.clear()
        return jsonify({"cleared": True})
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM analysis_history;")
        return jsonify({"cleared": True})
    finally:
        conn.close()


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"status": "error", "message": "That file is too large. Please upload an image under 12MB."}), 413


@app.errorhandler(500)
def internal_error(_e):
    log.exception("Unhandled server error")
    return jsonify({"status": "error", "message": "Something went wrong on our end. Please try again."}), 500


app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB upload limit

load_model()
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

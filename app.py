
from pathlib import Path
import sys
import json

import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
MODEL_PATH = PROJECT_DIR / "models" / "phishing_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
DATA_PATH = PROJECT_DIR / "data" / "sample_urls.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_extraction import extract_url_features, FEATURE_COLUMNS


app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "ui" / "templates"),
    static_folder=str(PROJECT_DIR / "ui" / "static"),
)


def load_model_package():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run: python src/train_models.py"
        )
    return joblib.load(MODEL_PATH)


def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": "Unknown",
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
    }


def load_dataset_summary():
    if not DATA_PATH.exists():
        return {
            "total": 0,
            "phishing": 0,
            "legitimate": 0,
            "https_ratio": 0,
        }

    df = pd.read_csv(DATA_PATH)
    total = len(df)
    phishing = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    legitimate = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    https_ratio = int(round((df["url"].astype(str).str.startswith("https://").mean()) * 100)) if total else 0

    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "https_ratio": https_ratio,
    }


@app.route("/")
def index():
    metadata = load_metadata()
    summary = load_dataset_summary()
    return render_template("index.html", metadata=metadata, summary=summary)


@app.route("/api/summary")
def summary_api():
    return jsonify({
        "metadata": load_metadata(),
        "summary": load_dataset_summary(),
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()

    if not url:
        return jsonify({
            "error": "Please enter a URL.",
        }), 400

    try:
        package = load_model_package()
        model = package["model"]
        feature_columns = package.get("feature_columns", FEATURE_COLUMNS)
        model_name = package.get("model_name", "Trained ML model")

        features = extract_url_features(url)
        X = pd.DataFrame([features])[feature_columns]

        prediction = int(model.predict(X)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            probability = float(proba[prediction])

        label = "Phishing" if prediction == 1 else "Legitimate"
        confidence = round(probability * 100, 2) if probability is not None else None

        return jsonify({
            "input_url": url,
            "normalised_url": features.get("url", url),
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
            "model_name": model_name,
            "features": features,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "hint": "Run `python src/train_models.py` first, then restart `python app.py`."
        }), 500


@app.route("/api/dataset")
def dataset():
    if not DATA_PATH.exists():
        return jsonify([])

    df = pd.read_csv(DATA_PATH)
    records = []
    for index, row in df.iterrows():
        url = str(row["url"])
        label = int(row["label"])
        records.append({
            "id": index + 1,
            "url": url,
            "label": "Phishing" if label == 1 else "Legitimate",
            "label_value": label,
            "length": len(url),
            "https": url.startswith("https://"),
        })

    return jsonify(records)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

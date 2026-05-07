"""
Train and save phishing URL detection models.

Run from the project root:
    python src/train_models.py

This script:
1. Loads data/sample_urls.csv
2. Extracts URL features
3. Trains several models
4. Saves the best model to models/phishing_model.joblib
5. Saves evaluation outputs to outputs/
"""

from pathlib import Path
import sys
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from feature_extraction import transform_urls_to_features, FEATURE_COLUMNS


DATA_PATH = PROJECT_DIR / "data" / "sample_urls.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"url", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = transform_urls_to_features(df["url"].astype(str).tolist())
    features = pd.DataFrame(feature_rows)
    features = features[FEATURE_COLUMNS]
    features["label"] = df["label"].values
    return features


def heuristic_baseline(row) -> int:
    score = 0

    if row["uses_https"] == 0:
        score += 1
    if row["has_ip_address"] == 1:
        score += 2
    if row["suspicious_word_count"] >= 1:
        score += 1
    if row["url_length"] > 75:
        score += 1
    if row["num_at_symbols"] > 0:
        score += 2
    if row["num_hyphens"] >= 3:
        score += 1

    return int(score >= 2)


def evaluate_predictions(name, y_true, predictions):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
    }


def main():
    df = load_dataset(DATA_PATH)
    features = build_feature_table(df)
    features.to_csv(OUTPUT_DIR / "extracted_features.csv", index=False)

    X = features.drop(columns=["label"])
    y = features["label"]

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ),
        "Simple Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=1200,
                random_state=42
            ))
        ]),
    }

    results = []

    heuristic_predictions = X_test.apply(heuristic_baseline, axis=1)
    results.append(evaluate_predictions("Heuristic Baseline", y_test, heuristic_predictions))

    best_model_name = None
    best_model = None
    best_f1 = -1

    report_text = []
    report_text.append("Heuristic Baseline\n")
    report_text.append(classification_report(y_test, heuristic_predictions, zero_division=0))
    report_text.append("\nConfusion Matrix:\n")
    report_text.append(str(confusion_matrix(y_test, heuristic_predictions)))
    report_text.append("\n\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(name, y_test, predictions)
        results.append(metrics)

        report_text.append(f"{name}\n")
        report_text.append(classification_report(y_test, predictions, zero_division=0))
        report_text.append("\nConfusion Matrix:\n")
        report_text.append(str(confusion_matrix(y_test, predictions)))
        report_text.append("\n\n")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("".join(report_text))

    model_package = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_model_name,
    }

    joblib.dump(model_package, MODEL_PATH)

    metadata = {
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(df)),
        "best_f1_score": float(best_f1),
        "label_mapping": {
            "0": "Legitimate",
            "1": "Phishing"
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plt.figure()
    plt.bar(results_df["model"], results_df["f1_score"])
    plt.title("Phishing Detection Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_score_comparison.png", dpi=150)

    print("\nModel Results")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

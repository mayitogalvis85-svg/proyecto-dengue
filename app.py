from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preparation import FEATURE_ORDER, prepare_batch, score_patient


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "outputs" / "modelos.joblib"
app = Flask(__name__)


def load_artifacts() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "No existe outputs/modelos.joblib. Ejecuta el notebook "
            "notebooks/01_analisis_y_modelado.ipynb antes de iniciar la aplicación."
        )
    import joblib

    return joblib.load(ARTIFACT_PATH)


def predict_features(features: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    artifacts = load_artifacts()
    results = {}
    for name, model in artifacts["models"].items():
        probabilities = model.predict_proba(features)[:, 1]
        labels = (probabilities >= 0.5).astype(int)
        results[name] = {
            "class": labels,
            "class_label": np.where(labels == 1, "Dengue complicado", "Dengue no complicado"),
            "probability": probabilities,
        }
    return results


def json_predictions(predictions: dict) -> dict:
    return {
        name: {
            "class": int(values["class"][0]),
            "class_label": str(values["class_label"][0]),
            "probability": float(values["probability"][0]),
        }
        for name, values in predictions.items()
    }


def evaluate_batch(y_true, y_pred, probabilities) -> dict:
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@app.get("/")
def index():
    artifacts = load_artifacts()
    return render_template(
        "index.html",
        metrics=artifacts.get("holdout_metrics", {}),
        cv_metrics=artifacts.get("cv_metrics", {}),
    )


@app.post("/api/predict")
def predict_individual():
    try:
        values = {feature: float(request.form[feature]) for feature in FEATURE_ORDER}
        features = pd.DataFrame([values], columns=FEATURE_ORDER)
        predictions = predict_features(features)
        score, score_label = score_patient(values)
        return jsonify({
            "predictions": json_predictions(predictions),
            "score": score,
            "score_label": score_label,
        })
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": f"Revisa los valores ingresados: {error}"}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict-batch")
def predict_batch():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Selecciona un archivo CSV o Excel."}), 400

    try:
        if uploaded.filename.lower().endswith((".xlsx", ".xls")):
            source = pd.read_excel(uploaded)
        else:
            source = pd.read_csv(uploaded)
        features, target = prepare_batch(source)
        predictions = predict_features(features)
        output = source.copy()
        response = {"rows": len(output), "has_target": target is not None, "models": {}}

        for name, values in predictions.items():
            short_name = "logistica" if name.startswith("Reg") else "rna"
            output[f"PREDICCION_{short_name.upper()}"] = values["class"]
            output[f"PROBABILIDAD_{short_name.upper()}"] = values["probability"]
            model_result = {
                "predictions": output[[f"PREDICCION_{short_name.upper()}", f"PROBABILIDAD_{short_name.upper()}"]]
                .head(100)
                .replace({np.nan: None})
                .to_dict(orient="records")
            }
            if target is not None:
                model_result["evaluation"] = evaluate_batch(target, values["class"], values["probability"])
            response["models"][name] = model_result

        response["columns"] = list(output.columns)
        response["preview"] = output.head(100).replace({np.nan: None}).to_dict(orient="records")
        return jsonify(response)
    except (ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

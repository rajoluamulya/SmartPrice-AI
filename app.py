"""Flask web application for the SmartPrice mobile price estimator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.predict import load_model, predict_price


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobile_price_pipeline.joblib"

app = Flask(__name__)


@app.get("/")
def home() -> str:
    """Render the mobile price estimation form."""
    return render_template("index.html")


@app.get("/health")
def health() -> tuple[Any, int] | Any:
    """Report whether the saved model is available and loadable."""
    try:
        load_model(str(MODEL_PATH))
    except (ImportError, OSError, TypeError, ValueError) as error:
        app.logger.exception("The saved model could not be loaded.")
        return jsonify(status="unavailable", error=str(error)), 503
    return jsonify(status="ok", model_available=True)


@app.post("/prediction")
def prediction() -> tuple[Any, int] | Any:
    """Estimate a phone price from a JSON payload or HTML form submission."""
    is_json = request.is_json
    features: object = request.get_json(silent=True) if is_json else request.form.to_dict()
    if not isinstance(features, dict):
        message = "Send a JSON object or submit the phone specification form."
        if is_json:
            return jsonify(error=message), 400
        return render_template("index.html", error=message), 400

    try:
        estimate = predict_price(features, model_path=MODEL_PATH)
    except ValueError as error:
        if is_json:
            return jsonify(error=str(error)), 400
        return render_template("index.html", error=str(error)), 400
    except (ImportError, OSError, TypeError) as error:
        app.logger.exception("Price prediction failed because the model is unavailable.")
        if is_json:
            return jsonify(error=str(error)), 503
        return render_template("index.html", error=str(error)), 503

    if is_json:
        return jsonify(predicted_price_inr=estimate)
    return render_template("index.html", predicted_price_inr=estimate)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))

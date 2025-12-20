from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from pathlib import Path
import numpy as np

# Flask app
app = Flask(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ensemble_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.joblib"

# -------------------------------
# Load model + metadata
# -------------------------------
model = load(MODEL_PATH)
meta = load(META_PATH)
FEATURES = meta["features"]

# Extract Random Forest feature importances from metadata
RF_FEATURE_IMPORTANCES = meta.get("rf_feature_importances", None)
if RF_FEATURE_IMPORTANCES is not None:
    RF_FEATURE_IMPORTANCES = RF_FEATURE_IMPORTANCES.tolist()  # Convert to list

# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def index():
    """Homepage"""
    return render_template("index.html")


@app.route("/patient-form")
def patient_form():
    """Form for entering patient details"""
    return render_template("patient_form.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction and show results"""

    # Collect input data from form
    input_data = []
    for f in FEATURES:
        val = request.form.get(f)
        try:
            input_data.append(float(val))
        except:
            input_data.append(0)

    # Convert to DataFrame
    input_df = pd.DataFrame([dict(zip(FEATURES, input_data))])

    # -------------------------------
    # Make prediction
    # -------------------------------
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    prediction = "Yes" if pred == 1 else "No"  # Yes = Cancer, No = No Cancer
    probability = f"{proba:.2%}"  # Display as percentage

    # -------------------------------
    # Render the prediction result page
    # -------------------------------
    return render_template(
        "prediction_result.html",
        prediction=prediction,
        probability=probability,
        feature_importances=RF_FEATURE_IMPORTANCES,
        feature_names=FEATURES
    )

@app.route("/data-info")
def data_info():
    """Dataset info page"""
    return render_template("data_info.html")


@app.route("/contact")
def contact():
    """Contact page"""
    return render_template("contact.html")


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
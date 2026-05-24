from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.joblib"

# -------------------------------
# Load model
# -------------------------------
model = load(MODEL_PATH)
meta = load(META_PATH)

FEATURES = meta["features"]
RF_FEATURE_IMPORTANCES = meta.get("rf_feature_importances", None)

if RF_FEATURE_IMPORTANCES is not None:
    RF_FEATURE_IMPORTANCES = list(RF_FEATURE_IMPORTANCES)

# -------------------------------
# Feature Importance Graph
# -------------------------------
def generate_feature_importance_chart(importances, feature_names):
    plt.figure()

    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = np.array(importances)[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]

    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()

    chart_path = os.path.join("static", "feature_importance.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

    return chart_path

# -------------------------------
# PDF Generator
# -------------------------------
def generate_pdf(prediction, probability, risk, advice, features, values):
    file_path = os.path.join("static", "report.pdf")

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Medical Prediction Report", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
    content.append(Paragraph(f"Probability: {probability}", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))
    content.append(Paragraph(f"Advice: {advice}", styles["Normal"]))

    content.append(Spacer(1, 20))
    content.append(Paragraph("Patient Details:", styles["Heading2"]))

    for f, v in zip(features, values):
        content.append(Paragraph(f"{f}:    {v}", styles["Normal"]))

    doc.build(content)

    return file_path

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/patient-form")
def patient_form():
    return render_template("patient_form.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():

    input_values = []
    

    for feature in FEATURES:
        value = request.form.get(feature)
        try:
            value = float(value)
        except:
            value = 0
        input_values.append(value)

    input_df = pd.DataFrame([dict(zip(FEATURES, input_values))])
    print(input_df)

    # Prediction
    probability = model.predict_proba(input_df)[0][1]
    print(probability)
    probability_percent = f"{probability*100:.2f}%"
    if(probability>=0.7):
        prediction="Cancer Detected"
    elif(probability>=0.5 and probability<=0.7):
        prediction="Almost confirmed as Cancer"
    else:
        probability="No Cancer"
    # -------------------------------
    # Risk Level
    # -------------------------------
    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # -------------------------------
    # Advice
    # -------------------------------
    if prediction == "Cancer Detected":
        advice = "Consult an oncologist immediately."
    else:
        advice = "Maintain a healthy lifestyle."

    # -------------------------------
    # Graph
    # -------------------------------
    chart_path = None
    if RF_FEATURE_IMPORTANCES is not None:
        chart_path = generate_feature_importance_chart(
            RF_FEATURE_IMPORTANCES,
            FEATURES
        )

    # -------------------------------
    # PDF
    # -------------------------------
    pdf_path = generate_pdf(
        prediction,
        probability_percent,
        risk,
        advice,
        FEATURES,
        input_values
    )

    return render_template(
        "prediction_result.html",
        prediction=prediction,
        probability=probability_percent,
        risk=risk,
        advice=advice,
        chart_path=chart_path,
        pdf_path=pdf_path
    )


@app.route("/data-info")
def data_info():
    return render_template("data_info.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
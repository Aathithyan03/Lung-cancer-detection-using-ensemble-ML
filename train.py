import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from scipy.stats import randint, uniform

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "lung_cancer_balanced.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load + Preprocess Dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Hyperparameter Tuning
# -------------------------------

# Logistic Regression
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))
])
log_params = {
    "clf__C": uniform(0.1, 10)
}
log_search = RandomizedSearchCV(log_pipeline, log_params, n_iter=20, cv=5, scoring="accuracy", random_state=42)
log_search.fit(X_train, y_train)
best_log = log_search.best_estimator_

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": randint(100, 800),
    "max_depth": randint(5, 50),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "bootstrap": [True, False]
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=30, cv=5, scoring="accuracy", random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=42))
])
svm_params = {
    "clf__C": uniform(0.1, 10),
    "clf__gamma": ["scale", "auto"]
}
svm_search = RandomizedSearchCV(svm_pipeline, svm_params, n_iter=20, cv=5, scoring="accuracy", random_state=42)
svm_search.fit(X_train, y_train)
best_svm = svm_search.best_estimator_

# -------------------------------
# Ensemble Voting Classifier
# -------------------------------
ensemble = VotingClassifier(
    estimators=[
        ("LogisticRegression", best_log),
        ("RandomForest", best_rf),
        ("SVM", best_svm)
    ],
    voting="soft",  # soft voting uses probabilities
    weights=[1, 4, 2]  # can tweak weights based on validation performance
)
ensemble.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred_ens = ensemble.predict(X_test)
y_proba_ens = ensemble.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_ens)
roc = roc_auc_score(y_test, y_proba_ens)

print("\n✅ Ensemble Voting Classifier Results:")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC: {roc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_ens))

# -------------------------------
# Save Model
# -------------------------------
dump(ensemble, MODEL_DIR / "ensemble_model.joblib")
dump({
    "features": list(X.columns),
    "target": "LUNG_CANCER"
}, MODEL_DIR / "model_meta.joblib")

print("\n🎉 Model trained and saved successfully!")
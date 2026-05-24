import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "lungs2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Safety check
df = df.dropna(subset=["LUNG_CANCER"])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset ready.\n")

# -------------------------------
# FAST Random Forest (No tuning)
# -------------------------------
print("Training model...")

rf = RandomForestClassifier(
    n_estimators=200,      # reduced
    max_depth=20,          # fixed value
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1              # use all CPU cores → faster
)

rf.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
print("\nEvaluating model...")

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n✅ Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.3f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
dump(rf, MODEL_DIR / "rf_model.joblib")

dump({
    "features": list(X.columns),
    "target": "LUNG_CANCER"
}, MODEL_DIR / "model_meta.joblib")

print("\n🎉 Model trained and saved successfully!")
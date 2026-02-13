import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find dataset at {DATA_PATH}. "
        f"Ensure you have ML_Assignment2/data/dataset.csv"
    )

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Basic checks
# -----------------------------
TARGET_COL = "stroke"
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {list(df.columns)}")

# Drop ID if present (not useful for learning)
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Split features/target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

print("Dataset loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:", y.value_counts().to_dict())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain/Test split done.")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# -----------------------------
# Identify numeric/categorical columns
# -----------------------------
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# -----------------------------
# Preprocessing pipelines
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# -----------------------------
# Metrics helper
# -----------------------------
def evaluate_model(name, model_pipeline, X_te, y_te):
    y_pred = model_pipeline.predict(X_te)

    y_prob = None
    if hasattr(model_pipeline, "predict_proba"):
        try:
            y_prob = model_pipeline.predict_proba(X_te)[:, 1]
        except Exception:
            y_prob = None

    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_te, y_pred),
    }

    # AUC only if prob available and both classes exist in y_te
    if y_prob is not None and len(set(y_te)) == 2:
        metrics["AUC"] = roc_auc_score(y_te, y_prob)
    else:
        metrics["AUC"] = None

    print(f"\n===== {name} Evaluation =====")
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: N/A")
        else:
            print(f"{k}: {v:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, digits=4))

    return metrics

# -----------------------------
# Train + Save Gradient Boosting
# -----------------------------
gb_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

print("\nTraining Gradient Boosting...")
gb_pipeline.fit(X_train, y_train)

gb_metrics = evaluate_model("Gradient Boosting", gb_pipeline, X_test, y_test)

gb_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
joblib.dump(gb_pipeline, gb_path)
print("Saved:", gb_path)

# -----------------------------
# Train + Save Decision Tree
# -----------------------------
dt_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

print("\nTraining Decision Tree...")
dt_pipeline.fit(X_train, y_train)

dt_metrics = evaluate_model("Decision Tree", dt_pipeline, X_test, y_test)

dt_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
joblib.dump(dt_pipeline, dt_path)
print("Saved:", dt_path)

# -----------------------------
# Summary table
# -----------------------------
summary = pd.DataFrame([
    {"Model": "Gradient Boosting", **gb_metrics},
    {"Model": "Decision Tree", **dt_metrics},
]).set_index("Model")

print("\n===== Metrics Summary =====")
print(summary)

import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="ML Assignment 2 - Stroke Prediction", layout="wide")
st.title("ML Assignment 2 - Stroke Prediction (Inference Only)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LABEL_COL = "stroke"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def discover_models(model_dir: str):
    """Return dict: {pretty_name: full_path} for all .pkl in model_dir."""
    model_map = {}
    if not os.path.isdir(model_dir):
        return model_map

    for fname in sorted(os.listdir(model_dir)):
        if fname.lower().endswith(".pkl"):
            pretty = os.path.splitext(fname)[0].replace("_", " ").title()
            model_map[pretty] = os.path.join(model_dir, fname)
    return model_map

def safe_predict_proba(model, X):
    """Return prob for class=1 if available; else None."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    return None


# -----------------------------
# Model dropdown (REQUIRED)
# -----------------------------
models = discover_models(MODEL_DIR)

if not models:
    st.error("No .pkl models found in the 'model/' folder. Run train_and_save.py to create them.")
    st.stop()

model_name = st.selectbox("Select Model (required)", list(models.keys()))
model_path = models[model_name]
model = load_model(model_path)
st.caption(f"Loaded: `{os.path.relpath(model_path, BASE_DIR)}`")


# -----------------------------
# CSV upload (REQUIRED)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload TEST CSV (required). If it contains a 'stroke' column, metrics/confusion matrix will be shown.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Data Preview")
st.dataframe(df.head())


# -----------------------------
# Split features/labels (if present)
# -----------------------------
has_labels = LABEL_COL in df.columns

if has_labels:
    y_true = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])
else:
    y_true = None
    X = df.copy()

# -----------------------------
# Run inference (NO FIT IN APP)
# -----------------------------
if st.button("Run Prediction"):
    y_pred = model.predict(X)
    y_prob = safe_predict_proba(model, X)

    results = X.copy()
    results["Predicted_Stroke"] = y_pred
    if y_prob is not None:
        results["Stroke_Probability"] = y_prob

    st.subheader("Prediction Results")
    st.dataframe(results)

    st.subheader("Summary")
    st.write("Total records:", len(results))
    st.write("Predicted stroke count:", int(pd.Series(y_pred).sum()))

    # -----------------------------
    # Metrics (REQUIRED)
    # -----------------------------
    st.subheader("Evaluation (required)")

    if not has_labels:
        st.warning(
            "No 'stroke' column found in the uploaded CSV, so metrics/confusion matrix cannot be computed.\n\n"
            "For grading, upload the test CSV that includes the true 'stroke' labels (if provided)."
        )
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1", f"{f1:.4f}")

        # -----------------------------
        # Confusion matrix / report (REQUIRED)
        # -----------------------------
        st.subheader("Confusion Matrix (required)")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df)

        st.subheader("Classification Report (required)")
        st.text(classification_report(y_true, y_pred, digits=4))

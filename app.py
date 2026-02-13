import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="ML Assignment 2 - Stroke Prediction", layout="wide")
st.title("ML Assignment 2 - Stroke Prediction (Inference App)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------- Helpers ----------
def list_model_files(model_dir: str) -> dict:
    """
    Returns a dict: {display_name: absolute_path}
    Display name comes from filename without extension, prettified.
    """
    model_map = {}
    if not os.path.isdir(model_dir):
        return model_map

    for fname in sorted(os.listdir(model_dir)):
        if fname.lower().endswith(".pkl"):
            display = os.path.splitext(fname)[0].replace("_", " ").title()
            model_map[display] = os.path.join(model_dir, fname)
    return model_map


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC only if probabilities are available and both classes exist
    if y_prob is not None and len(set(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUC"] = None
    return metrics


# ---------- Model selection dropdown (REQUIRED) ----------
model_paths = list_model_files(MODEL_DIR)

if not model_paths:
    st.error("No .pkl models found in the 'model/' folder. Please add at least one saved model.")
    st.stop()

model_choice = st.selectbox("Select Model (required)", list(model_paths.keys()))
model_path = model_paths[model_choice]
model = load_model(model_path)

st.caption(f"Loaded model file: `{os.path.relpath(model_path, BASE_DIR)}`")

# ---------- CSV upload (REQUIRED) ----------
uploaded_file = st.file_uploader(
    "Upload TEST CSV (required). If your CSV includes the true label column `stroke`, metrics will be shown.",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Data Preview")
st.dataframe(df.head())

# ---------- Handle optional label column ----------
LABEL_COL = "stroke"
has_label = LABEL_COL in df.columns

if has_label:
    y_true = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])
else:
    y_true = None
    X = df.copy()

# ---------- Predictions ----------
st.subheader("Prediction Results")

# Predict class
y_pred = model.predict(X)

# Predict probability (if available)
y_prob = None
if hasattr(model, "predict_proba"):
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None

results = X.copy()
results["Predicted_Stroke"] = y_pred
if y_prob is not None:
    results["Stroke_Probability"] = y_prob

st.dataframe(results)

# Summary (you already had this)
st.subheader("Summary")
st.write("Total records:", len(results))
st.write("Predicted stroke count:", int(results["Predicted_Stroke"].sum()))

# ---------- Metrics + Confusion Matrix / Classification Report (REQUIRED) ----------
st.subheader("Evaluation (required)")

if not has_label:
    st.warning(
        "Your uploaded CSV does NOT contain the true label column `stroke`, so metrics / confusion matrix cannot be computed.\n\n"
        "✅ For grading, upload the TEST CSV that includes the `stroke` column (true labels)."
    )
else:
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Metrics display (REQUIRED)
    st.markdown("### Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall", f"{metrics['Recall']:.4f}")
    col4.metric("F1", f"{metrics['F1']:.4f}")
    col5.metric("AUC", "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}")

    # Confusion matrix (REQUIRED: either this or report; we show both)
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df)

    # Classification report (REQUIRED alternative)
    st.markdown("### Classification Report")
    report = classification_report(y_true, y_pred, digits=4)
    st.text(report)

import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.title("Stroke Prediction App")

# =============================
# Model dropdown (REQUIRED)
# =============================
MODEL_DIR = "model"

models = {
    f.replace(".pkl", ""): os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.endswith(".pkl")
}

model_name = st.selectbox("Select Model", list(models.keys()))
model = joblib.load(models[model_name])

# =============================
# CSV Upload (REQUIRED)
# =============================
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):

        # =============================
        # Separate label if present
        # =============================
        if "stroke" in df.columns:
            y_true = df["stroke"]
            X = df.drop("stroke", axis=1)
        else:
            y_true = None
            X = df.copy()

        # =============================
        # Predictions
        # =============================
        preds = model.predict(X)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = None

        result_df = X.copy()
        result_df["Predicted_Stroke"] = preds

        if probs is not None:
            result_df["Stroke_Probability"] = probs

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        # =============================
        # Summary
        # =============================
        st.subheader("Summary")
        st.write("Total records:", len(result_df))
        st.write("Predicted stroke count:", int(preds.sum()))

        # =============================
        # Metrics (REQUIRED)
        # =============================
        if y_true is not None:

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds)
            rec = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)

            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"Precision: {prec:.4f}")
            st.write(f"Recall: {rec:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            # =============================
            # Confusion Matrix (REQUIRED)
            # =============================
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, preds)
            st.write(cm)

            st.subheader("Classification Report")
            st.text(classification_report(y_true, preds))
        else:
            st.warning("No 'stroke' column found. Metrics cannot be computed.")

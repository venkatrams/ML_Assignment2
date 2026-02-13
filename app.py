import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("Stroke Prediction App")
st.write("Upload a CSV file (features only) to generate predictions.")

MODEL_PATH = os.path.join("model", "gradient_boosting_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load model once (cached)
model = load_model()

uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Safe: drop target column if user uploads full dataset
    if "stroke" in df.columns:
        df = df.drop(columns=["stroke"])

    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        results = df.copy()
        results["Predicted_Stroke"] = preds
        results["Stroke_Probability"] = probs

        st.subheader("Prediction Results")
        st.dataframe(results)

        st.subheader("Summary")
        st.write("Total records:", len(results))
        st.write("Predicted stroke count:", int(results["Predicted_Stroke"].sum()))

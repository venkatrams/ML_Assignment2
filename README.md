# ML Assignment 2 – Classification Models

This repository contains the implementation of multiple
machine learning classification models and a Streamlit
web application for inference.

Due to class imbalance in the stroke dataset, accuracy alone is misleading.
Therefore, models were compared using Recall, F1, AUC, and MCC.
Different classifiers exhibited different trade-offs between sensitivity and precision.

Logistic Regression
Strong baseline, high AUC, but very conservative → low recall

Naive Bayes
Extremely high recall, suitable for screening, but poor precision

Decision Tree
Best F1-score, balanced detection, interpretable splits

Random Forest
Stable and robust, good AUC, but conservative on minority class

Gradient Boosting
Strong ranking ability (AUC), highest precision among ensemble models


# Stroke Prediction – ML Assignment 2

## How to run
1. Install requirements:
   pip install -r requirements.txt

2. Run Streamlit app:
   streamlit run app.py

3. Upload dataset CSV to get predictions.

## Model
- Gradient Boosting Classifier
- Full preprocessing pipeline included

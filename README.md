# ML Assignment 2 – Stroke Prediction Classification

This project implements multiple machine learning classification models for stroke prediction and deploys an interactive Streamlit web application for inference.

The complete end-to-end workflow includes:

- Data preprocessing (imputation, encoding, scaling)
- Model training and evaluation
- Comparison using multiple evaluation metrics
- Model persistence using joblib (.pkl)
- Streamlit web application (inference only)
- Deployment on Streamlit Community Cloud
- Execution verified on BITS Virtual Lab

---

## Dataset

The dataset used is the Kaggle Healthcare Stroke Dataset.

Target variable:
- `stroke` (0 = No Stroke, 1 = Stroke)

Due to class imbalance, accuracy alone is misleading.  
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- MCC

---

## Implemented Models

The following classification models were implemented and compared:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting

Observations:

- Logistic Regression: Strong baseline, high AUC, conservative predictions.
- Naive Bayes: Very high recall, suitable for screening.
- Decision Tree: Balanced detection and interpretable splits.
- Random Forest: Robust performance and good AUC.
- Gradient Boosting: Strong ranking performance and high precision.

The trained models are saved in the `model/` directory as `.pkl` files.

---

## Project Structure

```
ML_Assignment2/
│
├── data/
│   └── dataset.csv
│
├── model/
│   ├── gradient_boosting_model.pkl
│   └── decision_tree_model.pkl
│
├── train_and_save.py
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run Locally

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train and save models:
```
python train_and_save.py
```

3. Run Streamlit app:
```
streamlit run app.py
```

4. Upload test CSV file to generate predictions.

---

## Streamlit App Features

The deployed application includes:

- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics (Accuracy, Precision, Recall, F1)
- Confusion Matrix
- Classification Report
- Inference-only prediction (no training inside app)

Live App:
https://mlassignment2-5jgwjvubmcwpj4bs4jbruz.streamlit.app/

GitHub Repository:
https://github.com/venkatrams/ML_Assignment2/

---

## BITS Virtual Lab Execution

The project was executed in BITS Virtual Lab environment.  
Model loading was verified successfully using:

```
python -c "import joblib; joblib.load('model/gradient_boosting_model.pkl'); print('MODEL LOAD OK')"
```

---

## Author

Venkatram Sunkara  
BITS Pilani WILP – ML Assignment 2

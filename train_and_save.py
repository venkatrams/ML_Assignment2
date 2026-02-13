import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# ---- Load data ----
DATA_PATH = os.path.join("data", "dataset.csv")  # adjust if your filename differs
df = pd.read_csv(DATA_PATH)

# ---- Split features/target ----
if "stroke" not in df.columns:
    raise ValueError("Expected target column 'stroke' not found in dataset.")

X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Identify column types ----
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool"]).columns

# ---- Preprocessing pipelines ----
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---- Model pipeline ----
gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

gb_pipeline.fit(X_train, y_train)

# ---- Save model ----
os.makedirs("model", exist_ok=True)
out_path = os.path.join("model", "gradient_boosting_model.pkl")
joblib.dump(gb_pipeline, out_path)

print("Saved:", out_path)

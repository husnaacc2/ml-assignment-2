import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/breast_cancer.csv")

X = data.drop("target", axis=1)
y = data["target"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    return {
        "Accuracy": accuracy_score(y_te, y_pred),
        "AUC": roc_auc_score(y_te, y_prob),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "MCC": matthews_corrcoef(y_te, y_pred)
    }

# -----------------------------
# Train, Evaluate & Save
# -----------------------------
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    metrics = evaluate(model, X_test_scaled, y_test)
    results[name] = metrics

    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"model/{file_name}")

# -----------------------------
# Display Results
# -----------------------------
results_df = pd.DataFrame(results).T
print("\nFinal Evaluation Metrics:")
print(results_df.round(4))
# Save comparison results dynamically
results_df.to_csv("model/comparison_results.csv")

# -------------------------------------------------
# Save the exact test dataset used during training
# -------------------------------------------------
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv("model/fixed_test_data.csv", index=False)

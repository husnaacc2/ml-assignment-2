import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Machine Learning Classification Models")
st.write("Upload test data, select a model, and view evaluation results.")

# -------------------------------------------------
# Load scaler and models
# -------------------------------------------------
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# -------------------------------------------------
# Model selection
# -------------------------------------------------
selected_model_name = st.selectbox(
    "Select a classification model",
    list(models.keys())
)

# -------------------------------------------------
# File upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV file with target column)",
    type=["csv"]
)

# -------------------------------------------------
# Prediction and evaluation
# -------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = data.drop("target", axis=1)
        y_test = data["target"]

        X_test_scaled = scaler.transform(X_test)

        model = joblib.load(models[selected_model_name])
        y_pred = model.predict(X_test_scaled)

        # -----------------------------
        # Display metrics
        # -----------------------------
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
        col2.metric("Precision", round(precision_score(y_test, y_pred), 4))
        col3.metric("Recall", round(recall_score(y_test, y_pred), 4))

        st.metric("F1 Score", round(f1_score(y_test, y_pred), 4))

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
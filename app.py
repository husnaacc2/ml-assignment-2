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
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="2025AB05042_ML Classification App", layout="centered")

st.title("Machine Learning Model Evaluation Dashboard")
st.write("This application allows evaluation of trained classification models using either the built-in dataset or a user-provided dataset.")
st.write("You can choose to download the sample test dataset or upload your own CSV file (make sure it contains a 'target' column). You can also click the 'Load Default Test Data' button to load the default test dataset.")

# -------------------------------------------------
# Load Models and Scaler
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
# Dataset Section
# -------------------------------------------------
st.subheader("Test Dataset Selection")

if "data" not in st.session_state:
    st.session_state.data = None

col1, col2 = st.columns(2)

with col1:
    with open("model/fixed_test_data.csv", "rb") as f:
        st.download_button(
            label="Download Sample Test Data",
            data=f,
            file_name="test_dataset.csv",
            mime="text/csv"
        )

uploaded_file = st.file_uploader("Or upload your own dataset (CSV with target column)", type=["csv"])

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.success("Uploaded dataset loaded successfully.")

with col2:
    if st.button("Load Default Test Data"):
        st.session_state.data = pd.read_csv("model/fixed_test_data.csv")
        st.success("Default test dataset loaded successfully.")

data = st.session_state.data
# -------------------------------------------------
# Model Selection
# -------------------------------------------------
selected_model = st.selectbox(
    "Select a Trained Classification Model for Evaluation", list(models.keys())
)
# -------------------------------------------------
# Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Evaluation Section
# -------------------------------------------------
if data is not None:

    if "target" not in data.columns:
        st.error("Dataset must contain a 'target' column.")
    else:
        X_test = data.drop("target", axis=1)
        y_test = data["target"]

        X_scaled = scaler.transform(X_test)


        model = joblib.load(models[selected_model])
        y_pred = model.predict(X_scaled)

        # -----------------------------
        # Metric Calculations
        # -----------------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # AUC calculation (probability-based)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.0

        # -----------------------------
        # Display Metrics (2 rows × 3 columns)
        # -----------------------------
        st.subheader("Model Evaluation Metrics")

        # Row 1
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col2.metric("AUC", round(auc, 4))
        col3.metric("Precision", round(prec, 4))
        

        # Row 2
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", round(rec, 4))
        col5.metric("F1 Score", round(f1, 4))
        col6.metric("MCC", round(mcc, 4))
        

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)

        # -----------------------------
        # Overall Comparison Table
        # -----------------------------
        st.subheader("Comparative Performance of All Trained Models")

        comparison_df = pd.read_csv("model/comparison_results.csv")
        comparison_df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)

        # Highlight selected model
        def highlight_row(row):
            if row["Model"] == selected_model:
                return ["background-color: #2E8B57"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(comparison_df.style.apply(highlight_row, axis=1))

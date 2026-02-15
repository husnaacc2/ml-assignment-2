1. Problem Statement

The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification models on a real-world dataset. The task involves training different classifiers on the same dataset, evaluating their performance using standard classification metrics, and deploying the trained models through an interactive Streamlit web application. This assignment provides hands-on experience with the complete machine learning pipeline, including data preprocessing, model training, evaluation, and deployment.

2. Dataset Description

The dataset used for this assignment is the Breast Cancer Wisconsin (Diagnostic) dataset, obtained from the UCI Machine Learning Repository.

Number of instances: 569

Number of features: 30 numerical features

Target variable: Diagnosis of tumor

1 → Malignant

0 → Benign

The dataset consists of features computed from digitized images of breast mass biopsies. It satisfies the minimum requirements specified in the assignment, with more than 500 instances and more than 12 features. The raw dataset was preprocessed by converting categorical labels into numerical form and removing non-informative identifiers.

3. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented and evaluated on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

Each model was evaluated using the following metrics:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

Model Comparison Table:
                     Accuracy     AUC  Precision  Recall      F1     MCC
Logistic Regression    0.9649  0.9960     0.9750  0.9286  0.9512  0.9245
Decision Tree          0.9298  0.9246     0.9048  0.9048  0.9048  0.8492
KNN                    0.9561  0.9823     0.9744  0.9048  0.9383  0.9058
Naive Bayes            0.9211  0.9891     0.9231  0.8571  0.8889  0.8292
Random Forest          0.9737  0.9929     1.0000  0.9286  0.9630  0.9442
XGBoost                0.9737  0.9940     1.0000  0.9286  0.9630  0.9442

4. Observations on Model Performance:
Random Forest and XGBoost achieved the best overall performance, both recording
the highest accuracy (97.37%) and MCC (0.9442). Their perfect precision (1.000) indicates
zero false positives on the test set, demonstrating strong robustness and reliability. XGBoost
slightly outperformed Random Forest in AUC (0.9940 vs. 0.9929).

Logistic Regression also performed exceptionally well, achieving high accuracy
(96.49%) and the highest AUC (0.9960), indicating excellent class separability. Its balanced
precision (0.975) and recall (0.9286) resulted in strong F1 and MCC values, showing stable
generalization despite being a linear model.

KNN demonstrated competitive performance with accuracy of 95.61% and high precision
(0.9744). However, its slightly lower recall (0.9048) compared to ensemble models suggests
it may miss some positive instances near decision boundaries.

Decision Tree showed moderate performance, with accuracy of 92.98% and MCC of
0.8492. While precision and recall were balanced, the comparatively lower AUC (0.9246)
indicates reduced generalization capability, likely due to sensitivity to training data variations.

Naive Bayes achieved a strong AUC (0.9891) but recorded the lowest recall (0.8571) and
MCC (0.8292) among all models. This reflects the impact of its feature independence
assumption, which may not fully capture relationships between correlated features.

Overall, ensemble-based methods (Random Forest and XGBoost) provided the most
consistent and robust performance across all evaluation metrics, making them the
most suitable models for this classification task.
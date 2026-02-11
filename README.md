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

Model Comparison Table
                     Accuracy     AUC  Precision  Recall      F1     MCC
Logistic Regression    0.9649  0.9960     0.9750  0.9286  0.9512  0.9245
Decision Tree          0.9298  0.9246     0.9048  0.9048  0.9048  0.8492
KNN                    0.9561  0.9823     0.9744  0.9048  0.9383  0.9058
Naive Bayes            0.9211  0.9891     0.9231  0.8571  0.8889  0.8292
Random Forest          0.9737  0.9929     1.0000  0.9286  0.9630  0.9442
XGBoost                0.9737  0.9940     1.0000  0.9286  0.9630  0.9442

4. Observations on Model Performance
Logistic Regression	Logistic Regression achieved strong overall performance with an accuracy of 96.49% and a very high AUC of 0.996, indicating excellent class separability. The balanced precision (0.975) and recall (0.9286) resulted in a high F1 score and MCC, suggesting stable generalization and reliable predictions on unseen data.

Decision Tree	The Decision Tree model showed comparatively lower performance, with reduced accuracy (92.98%) and MCC (0.8492). While precision and recall were balanced, the drop in AUC indicates limited generalization, likely due to overfitting caused by its sensitivity to training data.

KNN	KNN achieved good accuracy (95.61%) and precision (0.9744), but slightly lower recall (0.9048) compared to ensemble models. This suggests that while KNN makes confident predictions, it may miss some positive cases, especially near class boundaries.

Naive Bayes	Naive Bayes demonstrated fast and efficient learning with a high AUC of 0.9891, but lower recall (0.8571) and MCC (0.8292). This performance reflects the impact of the strong feature independence assumption, which limits its effectiveness on correlated features.

Random Forest (Ensemble)	Random Forest achieved the highest overall performance, with accuracy of 97.37%, perfect precision (1.0), and strong MCC (0.9442). The ensemble approach effectively reduced variance and improved robustness, leading to consistent and reliable predictions.

XGBoost (Ensemble)	XGBoost matched Random Forest in accuracy (97.37%) and F1 score (0.9630), while maintaining a very high AUC of 0.994. Its gradient boosting framework allowed it to capture complex feature interactions efficiently, resulting in strong predictive performance across all metrics.
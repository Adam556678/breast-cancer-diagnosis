# Breast Cancer Diagnosis using Machine Learning Algorithms

## Project Overview

This project aims to build and evaluate several machine learning models to classify breast cancer tumors as either malignant (M) or benign (B). The workflow involves fetching the dataset from the UCI Machine Learning Repository, performing comprehensive data preprocessing, handling class imbalance, reducing dimensionality, and then training, tuning, and evaluating various classification models. The final evaluation is based on model accuracy and confusion matrices.

## Dataset

The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository.

-   **UCI ID:** 17
-   **Number of Instances:** 569
-   **Number of Features:** 30 real-valued features computed from digitized images of fine needle aspirates (FNA) of breast masses.
-   **Target Variable:** `Diagnosis` (M = malignant, B = benign)

Features describe characteristics of the cell nuclei, such as radius, texture, perimeter, area, smoothness, and more.

## Workflow

The project follows a systematic machine learning pipeline:

1.  **Data Loading & Exploration:** The dataset is fetched using the `ucimlrepo` library. Initial exploration is done to understand the data structure, feature names, and target variable distribution.

2.  **Handling Class Imbalance:** The initial dataset shows an imbalance with more benign samples (62.7%) than malignant ones (37.3%). To prevent model bias, the **Synthetic Minority Over-sampling Technique (SMOTE)** is applied to create a balanced dataset with a 50/50 class distribution.

3.  **Data Preprocessing:**
    *   **Label Encoding:** The categorical target variable `Diagnosis` ('M', 'B') is converted into numerical format (1, 0).
    *   **Feature Scaling:** `StandardScaler` from scikit-learn is used to scale the features, ensuring that all features contribute equally to the model's performance.

4.  **Dimensionality Reduction:**
    *   **Principal Component Analysis (PCA):** To reduce model complexity and potential overfitting, PCA is applied to reduce the 30 features down to the 3 most significant principal components. A scree plot is used to visualize the explained variance.

5.  **Model Training & Baseline Evaluation:**
    *   Four different classification models are trained on the preprocessed, reduced data:
        *   Logistic Regression
        *   Support Vector Classifier (SVC)
        *   Decision Tree Classifier
        *   XGBoost Classifier
    *   Initial performance is evaluated using 5-fold cross-validation to establish a baseline.

6.  **Hyperparameter Tuning:**
    *   `GridSearchCV` is used to find the optimal hyperparameters for Logistic Regression, Decision Tree, and XGBoost models to improve their predictive accuracy.

7.  **Final Model Evaluation:**
    *   The performance of the models before and after tuning is compared.
    *   **Confusion Matrices** are generated for each model on the test set to visualize their performance in terms of true positives, true negatives, false positives, and false negatives.

## Results

The models were evaluated based on their cross-validation scores and performance on the test set.

**Cross-Validation Accuracy (on training data):**

| Model                | Before Tuning | After Tuning | Improvement |
| -------------------- | ------------- | ------------ | ----------- |
| Logistic Regression  | 0.9159        | 0.8651       | -0.0508     |
| Decision Tree        | 0.9019        | 0.9019       | 0.0000      |
| XGBoost              | 0.9265        | 0.9299       | +0.0035     |

Hyperparameter tuning showed a slight improvement for the XGBoost model. The final evaluation using confusion matrices on the test data provides a clear picture of each model's ability to correctly classify tumors.

## Setup and Installation

To run this project, you need Python 3 and the following libraries. You can install them using pip:

```bash
pip install -r requirements.txt
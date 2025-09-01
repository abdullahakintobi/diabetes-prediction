# Diabetes Prediction Project Report

## Introduction

This project aims to predict whether a person has diabetes based on diagnostic health measures such as glucose level, BMI, age, blood pressure, and more. A **Support Vector Machine (SVM)** classifier with a linear kernel was used for the task. The dataset used is the **Pima Indians Diabetes Dataset**, a widely studied dataset for binary classification problems in healthcare.


## Dataset Overview

* **Source**: Pima Indians Diabetes Dataset (Kaggle/UCI)
* **Rows**: 768
* **Columns**: 9 (8 features + 1 target)

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target

* `Outcome`: 0 = Negative, 1 = Positive


## Methodology

1. **Data Loading**

   * The dataset was loaded from `data/diabetes.csv` using pandas.

2. **Preprocessing**

   * Features were separated from the target variable (`Outcome`).
   * Standardization was applied using `StandardScaler` to ensure features are on the same scale.

3. **Train-Test Split**

   * The dataset was split into **80% training** and **20% testing**, stratified by the target to preserve class balance.

4. **Model Training**

   * An **SVM classifier with linear kernel** was trained on the standardized training data.

5. **Evaluation**

   * Predictions were made on both training and testing sets.
   * Accuracy scores were computed using `accuracy_score`.


## Results

* **Training Accuracy**: \~78–80%
* **Test Accuracy**: \~75–77%

These results indicate that the model generalizes reasonably well without significant overfitting.


## Insights

* **Glucose levels** and **BMI** are among the strongest predictors of diabetes.
* Standardization significantly improves SVM performance by ensuring features contribute equally.
* The model achieves decent accuracy but leaves room for improvement through hyperparameter tuning or alternative algorithms.


## Limitations

* The dataset is relatively small (768 samples).
* Some features contain missing or zero values that could be better handled with imputation.
* Only accuracy was evaluated; other metrics such as precision, recall, and F1-score could provide deeper insights.


## Future Work

* Apply **hyperparameter tuning** (e.g., GridSearchCV) to optimize SVM performance.
* Experiment with additional models such as **Random Forest, XGBoost, and Neural Networks**.
* Incorporate **cross-validation** for more reliable evaluation.
* Expand evaluation metrics to include **precision, recall, F1-score, and ROC-AUC**.
* Build a **user-friendly interface** (Streamlit or Gradio) to make the model accessible to non-technical users.
* Explore deployment options with **FastAPI/Flask and Docker**.


## Conclusion

This project demonstrates how machine learning can be applied to predict diabetes from health metrics. While the baseline SVM model provides a reasonable starting point, further improvements in preprocessing, feature engineering, and model selection can significantly enhance predictive performance.
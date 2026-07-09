# Customer Churn Prediction using Machine Learning

A machine learning project that predicts whether a telecom customer will churn (leave the service) based on their account and service usage data. The project covers the full ML pipeline — from data cleaning and exploratory analysis to model training, hyperparameter tuning, and a working predictive system.

## Overview

Customer churn is one of the most important metrics for subscription-based businesses like telecom providers. This project uses the **Telco Customer Churn** dataset to build a classification model that flags customers likely to churn, so the business can take proactive retention action.

## 📂 Project Structure

```
Customer-Churn-Prediction_ML-main/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 customers, 21 columns)
├── notebooks/
│   └── Customer_Churn_Prediction_using_ML.ipynb   # End-to-end analysis & modeling notebook
├── reports/
│   └── ML_Customer_Churn_Prediction.pdf       # Project report / write-up
└── README.md
```

## 📊 Dataset

The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains 7,043 customer records with features including:

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account info:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Target:** `Churn` (Yes/No)

## 🔍 Workflow

1. **Data Collection** – Load the CSV into a pandas DataFrame.
2. **Exploratory Data Analysis (EDA)** – Inspect data types, check for missing values, and examine unique values across categorical columns.
3. **Data Preprocessing & Feature Extraction**
   - Dropped the `customerID` column (not predictive).
   - Fixed blank/whitespace entries in `TotalCharges` and cast it to `float`.
   - Encoded the target column (`Churn`) and all categorical features using `LabelEncoder`.
   - Identified class imbalance in the target variable.
4. **Feature Scaling & Train/Test Split** – Standardized features with `StandardScaler` and split data 80/20.
5. **Handling Class Imbalance** – Applied **SMOTE** (Synthetic Minority Oversampling) to balance the training data.
6. **Data Visualization** – Histograms, box plots, correlation heatmap, and count plots to understand feature distributions and outliers.
7. **Model Training & Hyperparameter Tuning**
   - Compared **Decision Tree**, **Random Forest**, and **K-Nearest Neighbors** using 5-fold cross-validation.
   - Tuned each model with `RandomizedSearchCV`.
8. **Model Selection** – Selected the best-performing model based on cross-validation accuracy.
9. **Performance Evaluation** – Evaluated the final model on the test set using accuracy, confusion matrix, and classification report.
10. **Model Persistence & Predictive System** – Saved the trained model (and feature names) with `pickle`, then reloaded it to run predictions on new customer input.

## 🏆 Results

**1. Default models, 5-fold CV on SMOTE-balanced training data:**

| Model | CV Accuracy |
|---|---|
| Decision Tree | 0.79 |
| Random Forest | 0.85 |
| K-Nearest Neighbors | 0.79 |

**2. After `RandomizedSearchCV` tuning, evaluated on the original (imbalanced) test set:**

| Model | Test Accuracy |
|---|---|
| Decision Tree | 77.4% |
| Random Forest | 79.2% |
| K-Nearest Neighbors | 70.0% |

**3. Best model selection** was based on each model's `RandomizedSearchCV` best CV score (not the table above) — **Random Forest** (`max_depth=20, n_estimators=500`) won with a CV accuracy of **0.85**.

**4. Final evaluation** of the selected Random Forest model was run on a *separately* SMOTE-balanced version of the test set (`X_test_smote`), giving:

- **Accuracy: 82.14%**

> Note: the 82.1% figure isn't directly comparable to the 79.2% in table 2 — they're measured on differently-balanced test sets (original vs. SMOTE-resampled), not a straightforward "before vs. after tuning" comparison.

Classification report (final model on the SMOTE-balanced test set):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No Churn (0) | 0.80 | 0.86 | 0.83 |
| Churn (1) | 0.84 | 0.79 | 0.82 |

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier), xgboost
- **Imbalanced data:** imbalanced-learn (SMOTE)
- **Model persistence:** pickle

## ▶️ How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
2. Open the notebook:
   ```bash
   jupyter notebook notebooks/Customer_Churn_Prediction_using_ML.ipynb
   ```
3. Run all cells in order — the notebook loads the dataset from `data/`, trains the models, saves the best model as `best_model.pkl`, and demonstrates a sample prediction.

## 📄 Report

A detailed write-up of the methodology and findings is available in [`reports/ML_Customer_Churn_Prediction.pdf`](reports/ML_Customer_Churn_Prediction.pdf).

## 🔮 Future Improvements

- Try additional models (XGBoost is imported but not yet benchmarked against the other three).
- Perform more rigorous feature engineering (e.g., tenure buckets, interaction terms).
- Deploy the model behind a simple API (e.g., FastAPI/Flask) for real-time predictions.
- Add SHAP/feature-importance-based explainability for business stakeholders.

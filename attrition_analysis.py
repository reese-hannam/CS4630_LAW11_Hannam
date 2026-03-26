"""
CS 4/5630 - Week 11: Supervised Machine Learning
Predicting Employee Attrition Using Workplace and Performance Factors
Dataset: IBM HR Analytics Employee Attrition & Performance
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

os.makedirs("output", exist_ok=True)

# Tee output to both terminal and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log = open("output/results.txt", "w")
sys.stdout = Tee(sys.__stdout__, log)

# =============================================================================
# STEP 1: Load the Dataset
# =============================================================================

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv.xls",
    encoding="utf-8-sig"
)

print("=" * 55)
print("STEP 1: Dataset Overview")
print("=" * 55)
print(f"Dataset shape: {df.shape}")
print(f"\nAttrition counts:\n{df['Attrition'].value_counts()}")
print(f"\nAttrition rate: {(df['Attrition'] == 'Yes').mean():.1%}")

# =============================================================================
# STEP 2: Address Class Imbalance
# =============================================================================

print("\n" + "=" * 55)
print("STEP 2: Preprocessing & Class Imbalance")
print("=" * 55)

# drop columns with no predictive value
df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], inplace=True)

# encode categorical columns — use a separate encoder per column
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Attrition")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# encode target
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print(f"Training set before SMOTE: {dict(y_train.value_counts().sort_index())}")
print(f"Training set after SMOTE:  {dict(pd.Series(y_train_bal).value_counts().sort_index())}")
print(f"Test set (unchanged):      {dict(y_test.value_counts().sort_index())}")

# =============================================================================
# STEP 3: Exploratory Data Analysis (summary stats only — see visualizations.py)
# =============================================================================

print("\n" + "=" * 55)
print("STEP 3: EDA Summary")
print("=" * 55)

num_cols = ["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
            "YearsAtCompany", "DistanceFromHome", "NumCompaniesWorked",
            "TotalWorkingYears", "EnvironmentSatisfaction"]

print("\nAttrition rate by key features:")
for col in ["JobSatisfaction", "WorkLifeBalance", "OverTime"]:
    rates = df.groupby(col)["Attrition"].mean() * 100
    print(f"\n  {col}:\n{rates.to_string()}")

print(f"\nCorrelations with Attrition (sorted low to high):")
corr = df[num_cols + ["Attrition"]].corr()["Attrition"].drop("Attrition").sort_values()
print(corr.to_string())

print(f"\nStrongest positive correlations with Attrition:")
print(corr[corr > 0].sort_values(ascending=False).to_string())

print(f"\nStrongest negative correlations with Attrition:")
print(corr[corr < 0].sort_values().to_string())

# =============================================================================
# STEP 4: Build and Tune Classifiers
# =============================================================================

print("\n" + "=" * 55)
print("STEP 4: Model Training & Hyperparameter Tuning")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Random Forest ---
# not using class_weight="balanced" here since SMOTE already balanced the training data
print("\nTuning Random Forest...")
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
    cv=cv, scoring="f1", n_jobs=-1
)
rf_gs.fit(X_train_bal, y_train_bal)
print(f"  Best params: {rf_gs.best_params_}")

# --- KNN ---
print("\nTuning K-Nearest Neighbors...")
knn_gs = GridSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
    cv=cv, scoring="f1", n_jobs=-1
)
knn_gs.fit(X_train_bal, y_train_bal)
print(f"  Best params: {knn_gs.best_params_}")

# --- Naive Bayes ---
# GaussianNB has no hyperparameters to tune, used as a probabilistic baseline
print("\nTraining Naive Bayes (GaussianNB)...")
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
print("  No hyperparameters to tune for GaussianNB")

# =============================================================================
# STEP 5: Evaluate Model Performance
# =============================================================================

print("\n" + "=" * 55)
print("STEP 5: Model Evaluation")
print("=" * 55)

models = {
    "Random Forest": rf_gs.best_estimator_,
    "KNN": knn_gs.best_estimator_,
    "Naive Bayes": nb_model,
}

results = {}
for name, model in models.items():
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    # RF uses a lower threshold to improve recall on the minority class
    if name == "Random Forest":
        y_pred = (y_prob >= 0.3).astype(int)
    else:
        y_pred = model.predict(X_test_sc)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "auc": auc,
        "cm": confusion_matrix(y_test, y_pred),
    }
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"  ROC-AUC: {auc:.4f}")

# summary table
print("\n" + "=" * 55)
print("MODEL SUMMARY (minority class: Attrition = Yes)")
print("=" * 55)
print(f"{'Model':<16} {'Precision':>9} {'Recall':>7} {'F1':>7} {'AUC':>7}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<16} {res['precision']:>9.3f} {res['recall']:>7.3f} "
          f"{res['f1']:>7.3f} {res['auc']:>7.3f}")

best_f1 = max(results, key=lambda m: results[m]["f1"])
best_auc = max(results, key=lambda m: results[m]["auc"])
print(f"\nBest model by F1-Score:  {best_f1}")
print(f"Best model by ROC-AUC:   {best_auc}")
print("\nNote: F1 and AUC can point to different models on imbalanced data.")
print("Note: See visualizations.py to generate all plots.")


# test threshold 0.3 on KNN and Naive Bayes for comparison
print("\n" + "=" * 55)
print("THRESHOLD 0.65 TEST — KNN AND NAIVE BAYES")
print("=" * 55)

for name in ["KNN", "Naive Bayes"]:
    y_pred_tuned = (results[name]["y_prob"] >= 0.65).astype(int)
    print(f"\n{name} at threshold 0.65:")
    print(classification_report(y_test, y_pred_tuned, target_names=["No", "Yes"]))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")

# save results for visualizations.py
import pickle
with open("output/results.pkl", "wb") as f:
    pickle.dump({
        "results": results,
        "X_test_sc": X_test_sc,
        "y_test": y_test,
        "X_columns": list(X.columns),
        "rf_model": rf_gs.best_estimator_,
        "encoders": encoders,
    }, f)

sys.stdout = sys.__stdout__
log.close()
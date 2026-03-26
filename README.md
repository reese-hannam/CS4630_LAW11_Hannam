# CS4630 Week 11 - Predicting Employee Attrition
Reese Hannam

## Dataset
IBM HR Analytics Employee Attrition & Performance
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Download the dataset and place it in the root directory before running.

## How to Run
1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib

2. Run the main script
python attrition_analysis.py

3. Generate visualizations
python visualizations.py

## Output
All results are saved to the output/ folder.
- results.txt - full model evaluation output
- results.pkl - saved model results for visualizations
- class_imbalance.png — shows the class distribution before and after SMOTE was applied
- roc_curves.png — compares each model's ability to distinguish attrition cases from non-attrition cases
- confusion_matrices.png — shows how many attrition cases each model correctly caught versus missed
- model_comparison.png — side by side comparison of precision, recall, F1, and AUC across all three models
- feature_importance.png — shows which employee features the Random Forest relied on most when making predictions


## Models
Three classifiers were trained and evaluated on the minority class (Attrition = Yes):
- Random Forest (threshold 0.3)
- K-Nearest Neighbors
- Naive Bayes

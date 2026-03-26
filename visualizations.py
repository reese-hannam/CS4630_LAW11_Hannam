"""
CS 4/5630 - Week 11: Supervised Machine Learning
visualizations.py — generates plots for the presentation
run after main.py (needs output/results.pkl)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# load results saved by main.py
with open("output/results.pkl", "rb") as f:
    data = pickle.load(f)

results   = data["results"]
y_test    = data["y_test"]
X_columns = data["X_columns"]
rf_model  = data["rf_model"]

# =============================================================================
# 1. Class Imbalance — Before / After SMOTE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Class Imbalance: Before and After SMOTE", fontsize=13, fontweight="bold")

before = {"No (0)": 986, "Yes (1)": 190}
after  = {"No (0)": 986, "Yes (1)": 986}

for ax, counts, title in zip(axes, [before, after], ["Before SMOTE", "After SMOTE"]):
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["steelblue", "coral"], width=0.5)
    ax.set_title(title)
    ax.set_ylabel("Sample Count")
    ax.set_ylim(0, 1150)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(int(bar.get_height())), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("output/class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/class_imbalance.png")

# =============================================================================
# 2. ROC Curves
# =============================================================================

model_colors = ["blue", "green", "red"]

fig, ax = plt.subplots(figsize=(7, 6))

for (name, res), color in zip(results.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})",
            color=color, linewidth=2)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig("output/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/roc_curves.png")

# =============================================================================
# 3. Confusion Matrices
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = res["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_title(name, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No", "Pred: Yes"])
    ax.set_yticklabels(["Act: No", "Act: Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("output/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/confusion_matrices.png")

# =============================================================================
# 4. Model Comparison — Precision / Recall / F1 / AUC
# =============================================================================

metrics = ["precision", "recall", "f1", "auc"]
labels  = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
x       = np.arange(len(labels))
width   = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, color) in enumerate(zip(results.keys(), model_colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison (Minority Class: Attrition = Yes)",
             fontweight="bold")
ax.legend()
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("output/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/model_comparison.png")

# =============================================================================
# 5. Feature Importance — Random Forest Top 15
# =============================================================================

importances = rf_model.feature_importances_
indices     = np.argsort(importances)[-15:]
feat_names  = [X_columns[i] for i in indices]
feat_vals   = importances[indices]

# highlight the top 5 in orange, rest in blue
bar_colors = ["steelblue"] * 10 + ["coral"] * 5

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_names, feat_vals, color=bar_colors)
ax.set_xlabel("Feature Importance")
ax.set_xlim(0, max(feat_vals) * 1.15)
ax.set_title("Random Forest — Top 15 Feature Importances", fontweight="bold")

for i, val in enumerate(feat_vals):
    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)

# manual legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="coral",     label="Top 5"),
    Patch(color="steelblue", label="6–15"),
], fontsize=9)

plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/feature_importance.png")

print("\nDone. All plots saved to output/")
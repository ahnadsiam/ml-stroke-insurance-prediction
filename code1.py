"""
HIT391 Assignment 2 - Task 1: Classification
Dataset: D2 - Healthcare Stroke Dataset
File: 21.py
Student: Mohamad Ahmed (s374721)
"""

# ─────────────────────────────────────────────────────────────────
# STEP 1: Loading Data, Data Pre-processing, EDA
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report,
                              roc_curve)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('charts', exist_ok=True)

# Load data
df = pd.read_csv('/home/claude/dataset/Datasets/D2/healthcare-dataset-stroke-data.csv')
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df['stroke'].value_counts())

# ── EDA ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['stroke'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'salmon'], edgecolor='black')
axes[0].set_title('Stroke Class Distribution')
axes[0].set_xlabel('Stroke (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

# Age distribution by stroke
df.groupby('stroke')['age'].plot(kind='hist', bins=20, alpha=0.6, ax=axes[1])
axes[1].set_title('Age Distribution by Stroke')
axes[1].set_xlabel('Age')
axes[1].legend(['No Stroke', 'Stroke'])
plt.tight_layout()
plt.savefig('charts/d2_eda.png', dpi=150)
plt.close()
print("\n[Saved] charts/d2_eda.png")

# Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(9, 7))
num_cols = df.select_dtypes(include=np.number).drop(columns=['id'])
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap - D2 Stroke Dataset')
plt.tight_layout()
plt.savefig('charts/d2_corr.png', dpi=150)
plt.close()
print("[Saved] charts/d2_corr.png")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Feature Engineering, Train/Test Split
# ─────────────────────────────────────────────────────────────────
df = df.drop(columns=['id'])

# Fill missing BMI with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\n=== AFTER PREPROCESSING ===")
print(df.describe())

X = df.drop(columns=['stroke'])
y = df['stroke']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"\nAfter SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────
# STEP 3: Apply at least 2 Classification Algorithms
# ─────────────────────────────────────────────────────────────────

# ── Algorithm 1: Logistic Regression ──
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# ── Algorithm 2: Random Forest ──
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────────────────────────
# STEP 4: Evaluation Metrics (at least 2 per algorithm)
# ─────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    print(f"\n── {name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['No Stroke', 'Stroke']))
    return acc, f1, auc

print("\n=== EVALUATION METRICS ===")
acc_lr, f1_lr, auc_lr = evaluate("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
acc_rf, f1_rf, auc_rf = evaluate("Random Forest",       y_test, y_pred_rf, y_prob_rf)

# ─────────────────────────────────────────────────────────────────
# STEP 5: Comparing Results
# ─────────────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model':    ['Logistic Regression', 'Random Forest'],
    'Accuracy': [acc_lr, acc_rf],
    'F1 Score': [f1_lr, f1_rf],
    'ROC-AUC':  [auc_lr, auc_rf]
})
print("\n=== COMPARISON TABLE ===")
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
colors  = ['steelblue', 'darkorange']
for i, metric in enumerate(metrics):
    axes[i].bar(results['Model'], results[metric], color=colors, edgecolor='black')
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1.05)
    axes[i].set_ylabel('Score')
    for j, v in enumerate(results[metric]):
        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
plt.suptitle('Classification Model Comparison - D2 Stroke', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/d2_comparison.png', dpi=150)
plt.close()
print("\n[Saved] charts/d2_comparison.png")

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', color='steelblue')
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})',       color='darkorange')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Stroke Classification')
ax.legend(); plt.tight_layout()
plt.savefig('charts/d2_roc.png', dpi=150)
plt.close()
print("[Saved] charts/d2_roc.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y_pred, title in zip(axes,
    [y_pred_lr, y_pred_rf],
    ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    ax.set_title(f'Confusion Matrix – {title}')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('charts/d2_cm.png', dpi=150)
plt.close()
print("[Saved] charts/d2_cm.png")

# ─────────────────────────────────────────────────────────────────
# STEP 6: Fine-Tune Best Algorithm (Random Forest)
# ─────────────────────────────────────────────────────────────────
print("\n=== STEP 6: HYPERPARAMETER TUNING (Random Forest) ===")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best Params : {grid.best_params_}")
print(f"Best CV AUC : {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
y_prob_tuned = best_rf.predict_proba(X_test)[:, 1]
print(f"\nTuned Random Forest:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred_tuned):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob_tuned):.4f}")

# Feature importance
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Feature Importances – Tuned Random Forest')
ax.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('charts/d2_feature_importance.png', dpi=150)
plt.close()
print("[Saved] charts/d2_feature_importance.png")

print("\n=== Task 1 Complete ===")

"""
HIT391 Assignment 2 - Task 2: Regression
Dataset: D3 - Medical Insurance Costs
File: 32.py
Student: Mohamad Ahmed (s374721)
"""

# ─────────────────────────────────────────────────────────────────
# STEP 1: Loading Data, Data Pre-processing, EDA
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('charts', exist_ok=True)

# Load data
df = pd.read_csv('insurance.csv')  # place this script in the same folder as the CSV
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:")
print(df.describe())

# ── EDA ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Target distribution
axes[0].hist(df['charges'], bins=30, color='steelblue', edgecolor='black')
axes[0].set_title('Distribution of Insurance Charges')
axes[0].set_xlabel('Charges (USD)')
axes[0].set_ylabel('Frequency')

# Charges by smoker status
df.boxplot(column='charges', by='smoker', ax=axes[1],
           boxprops=dict(color='steelblue'),
           medianprops=dict(color='red'))
axes[1].set_title('Charges by Smoker Status')
axes[1].set_xlabel('Smoker')
axes[1].set_ylabel('Charges')
plt.suptitle('')
plt.tight_layout()
plt.savefig('charts/d3_eda.png', dpi=150)
plt.close()
print("\n[Saved] charts/d3_eda.png")

# Scatter plots: age & BMI vs charges, coloured by smoker
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = df['smoker'].map({'yes': 'salmon', 'no': 'steelblue'})
axes[0].scatter(df['age'], df['charges'], c=colors, alpha=0.5, edgecolors='none')
axes[0].set_title('Age vs Charges (red=smoker)')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Charges')
axes[1].scatter(df['bmi'], df['charges'], c=colors, alpha=0.5, edgecolors='none')
axes[1].set_title('BMI vs Charges (red=smoker)')
axes[1].set_xlabel('BMI'); axes[1].set_ylabel('Charges')
plt.tight_layout()
plt.savefig('charts/d3_scatter.png', dpi=150)
plt.close()
print("[Saved] charts/d3_scatter.png")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Feature Engineering, Train/Test Split
# ─────────────────────────────────────────────────────────────────
le = LabelEncoder()
for col in ['sex', 'smoker', 'region']:
    df[col] = le.fit_transform(df[col])

# Interaction feature: bmi × smoker (known strong predictor)
df['bmi_smoker'] = df['bmi'] * df['smoker']

print("\n=== AFTER ENCODING ===")
print(df.head())

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap - D3 Insurance Dataset')
plt.tight_layout()
plt.savefig('charts/d3_corr.png', dpi=150)
plt.close()
print("[Saved] charts/d3_corr.png")

X = df.drop(columns=['charges'])
y = df['charges']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────
# STEP 3: Apply at least 2 Regression Algorithms
# ─────────────────────────────────────────────────────────────────

# ── Algorithm 1: Linear Regression ──
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ── Algorithm 2: Random Forest Regressor ──
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)

# ─────────────────────────────────────────────────────────────────
# STEP 4: Evaluation Metrics (at least 2 per algorithm)
# ─────────────────────────────────────────────────────────────────
def evaluate_reg(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n── {name} ──")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  R²   : {r2:.4f}")
    return rmse, mae, r2

print("\n=== EVALUATION METRICS ===")
rmse_lr,  mae_lr,  r2_lr  = evaluate_reg("Linear Regression",       y_test, y_pred_lr)
rmse_rfr, mae_rfr, r2_rfr = evaluate_reg("Random Forest Regressor", y_test, y_pred_rfr)

# ─────────────────────────────────────────────────────────────────
# STEP 5: Comparing Results
# ─────────────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE':  [rmse_lr, rmse_rfr],
    'MAE':   [mae_lr,  mae_rfr],
    'R²':    [r2_lr,   r2_rfr]
})
print("\n=== COMPARISON TABLE ===")
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['steelblue', 'darkorange']
for ax, metric in zip(axes, ['RMSE', 'MAE', 'R²']):
    vals = results[metric]
    ax.bar(results['Model'], vals, color=colors, edgecolor='black')
    ax.set_title(metric)
    ax.set_ylabel('Score')
    for j, v in enumerate(vals):
        ax.text(j, v * 1.01, f'{v:,.0f}' if metric != 'R²' else f'{v:.3f}',
                ha='center', fontsize=9)
plt.suptitle('Regression Model Comparison - D3 Insurance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/d3_comparison.png', dpi=150)
plt.close()
print("[Saved] charts/d3_comparison.png")

# Actual vs Predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_pred, title, color in zip(axes,
    [y_pred_lr, y_pred_rfr],
    ['Linear Regression', 'Random Forest Regressor'],
    ['steelblue', 'darkorange']):
    ax.scatter(y_test, y_pred, alpha=0.4, color=color, edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel('Actual Charges')
    ax.set_ylabel('Predicted Charges')
    ax.set_title(f'{title}: Actual vs Predicted')
plt.tight_layout()
plt.savefig('charts/d3_actual_vs_pred.png', dpi=150)
plt.close()
print("[Saved] charts/d3_actual_vs_pred.png")

# ─────────────────────────────────────────────────────────────────
# STEP 6: Fine-Tune Best Algorithm (Random Forest Regressor)
# ─────────────────────────────────────────────────────────────────
print("\n=== STEP 6: HYPERPARAMETER TUNING (Random Forest Regressor) ===")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best Params : {grid.best_params_}")
print(f"Best CV R²  : {grid.best_score_:.4f}")

best_rfr = grid.best_estimator_
y_pred_tuned = best_rfr.predict(X_test)
print(f"\nTuned Random Forest Regressor:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_tuned)):,.2f}")
print(f"  MAE  : {mean_absolute_error(y_test, y_pred_tuned):,.2f}")
print(f"  R²   : {r2_score(y_test, y_pred_tuned):.4f}")

# Feature importance
feat_imp = pd.Series(best_rfr.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind='bar', ax=ax, color='darkorange', edgecolor='black')
ax.set_title('Feature Importances – Tuned Random Forest Regressor')
ax.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('charts/d3_feature_importance.png', dpi=150)
plt.close()
print("[Saved] charts/d3_feature_importance.png")

print("\n=== Task 2 Complete ===")

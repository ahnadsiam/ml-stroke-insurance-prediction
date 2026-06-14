# ML Pipelines: Stroke Prediction & Insurance Cost Estimation

Two end-to-end machine learning pipelines applied to real-world healthcare datasets, covering classification and regression tasks with full preprocessing, model training, evaluation, and hyperparameter tuning.

---

## Task 1: Stroke Risk Classification

**Dataset:** 5,110 patient records | 11 features (age, BMI, glucose level, hypertension, smoking status, etc.)

**Problem:** Binary classification — predict whether a patient is at risk of stroke.

**Approach:**
- Imputed missing BMI values using column median
- Label-encoded categorical features; scaled with StandardScaler
- Applied SMOTE to address severe class imbalance (only 4.9% positive cases)
- Trained and compared Logistic Regression and Random Forest (100 trees)
- Tuned Random Forest using GridSearchCV (max_depth, min_samples_split, n_estimators)

**Results:**

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8118 | 0.8153 | 0.8951 |
| Random Forest | 0.9460 | 0.9469 | 0.9889 |
| Tuned Random Forest | 0.9445 | 0.9455 | 0.9893 |

**Top predictors:** Age, average glucose level, BMI

---

## Task 2: Medical Insurance Cost Prediction

**Dataset:** 1,338 records | 6 features (age, sex, BMI, children, smoker status, region)

**Problem:** Regression — predict annual insurance charges (USD).

**Approach:**
- Engineered interaction feature `bmi_smoker` (BMI x smoker status) to capture compounding health risk
- Label-encoded categorical columns; scaled with StandardScaler
- Trained Linear Regression (baseline) and Random Forest Regressor (100 trees)
- Tuned with GridSearchCV (max_depth=10, min_samples_split=5)

**Results:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 4,577 | 2,770 | 0.8650 |
| Random Forest | 4,555 | 2,510 | 0.8663 |
| Tuned Random Forest | 4,516 | 2,508 | 0.8687 |

**Top predictors:** Smoker status, bmi_smoker interaction feature, age

---

## Tech Stack

- Python
- scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- imbalanced-learn (SMOTE)

---

## Key Takeaways

- Random Forest substantially outperformed Logistic Regression on the stroke dataset (ROC-AUC: 0.989 vs 0.895)
- The engineered `bmi_smoker` interaction term significantly improved insurance cost predictions
- SMOTE was critical for handling class imbalance in the stroke dataset — without it, the model would have been heavily biased toward the majority class

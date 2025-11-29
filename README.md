# Predicting-Loan-Payback---Kaggle
Building a loan payback prediction model for a kaggle competition using CAT boost classifier
# Loan Default Probability Prediction

## Project Overview

This project predicts the probability of a loan being paid back using both numerical and categorical features. The model outputs **probability scores** for each loan applicant, allowing financial institutions to assess loan risk effectively.

The model is evaluated using **ROC-AUC**, which measures the ranking performance of predicted probabilities.

---

## Initial Approach

The first implementation used a **basic CatBoost classifier** with minimal preprocessing.

- **Initial ROC-AUC:** 0.79347
- Limitations:
  - Inconsistent label encoding for categorical variables
  - Presence of outliers in numeric features
  - No feature scaling or normalization applied
  - Default hyperparameters were under-optimized

---

## Evolved Approach

The model was improved iteratively with the following steps:

### 1. Data Preprocessing

- **Categorical Variables:**
  - Converted to `category` type in pandas
  - Ensured consistent mapping between train and test sets using label encoding
  - Unseen categories in test data were mapped to `-1`
- **Outlier Handling:**
  - Used **IQR (Interquartile Range) method** and optionally **Isolation Forest**
  - Stabilized numeric feature distributions
- **Normalization (Optional):**
  - Applied `MinMaxScaler` or `StandardScaler` after outlier removal
  - CatBoost does not require normalization, but it can help with other models

---

### 2. Model Selection and Training

- Switched to an **optimized CatBoostClassifier** on GPU for speed
- Used **ROC-AUC** as the main evaluation metric
- Implemented **early stopping** to prevent overfitting

**Optimized CatBoost Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| iterations | 3200 | Number of boosting rounds |
| learning_rate | 0.02–0.025 | Step size for gradient boosting |
| depth | 8 | Maximum tree depth |
| l2_leaf_reg | 3–4 | L2 regularization to prevent overfitting |
| bootstrap_type | Poisson | Compatible with GPU and subsampling |
| subsample | 0.8 | Fraction of rows sampled per iteration |
| bagging_temperature | 1.0 | Adds randomness for better generalization |
| task_type | GPU | Enables GPU acceleration |
| eval_metric | AUC | Optimizing ROC-AUC directly |
| loss_function | Logloss | Standard binary classification loss |
| random_seed | 51 | Reproducibility |
| early_stopping_rounds | 200 | Stops training when validation AUC stops improving |

---

### 3. Feature Engineering

- Verified feature importance via `model.get_feature_importance()`
- Removed or adjusted noisy features
- Created interaction features when applicable

---

### 4. Model Evaluation

- Used **ROC-AUC** as the main metric
- Visualized **ROC curves** for validation
- Applied the final model to test set to predict probabilities for each `id`

---

## Final Outcome

- **Final ROC-AUC:** 0.92292  
- **Improvement:** +0.12945 from the initial model

**Key Improvements:**
1. Outlier removal stabilized numeric features
2. Proper categorical encoding ensured consistency
3. GPU-enabled CatBoost with tuned hyperparameters improved ranking
4. Early stopping and feature engineering reduced overfitting

---

## Next Steps / Potential Improvements

- **Hyperparameter Tuning:** Use Optuna or Bayesian optimization
- **Feature Engineering:** Derive additional interaction features
- **Ensemble Models:** Combine CatBoost with XGBoost or LightGBM
- **Automated Contamination Selection:** Optimize Isolation Forest contamination automatically

---

## Workflow Diagram

graph TD
    A["Raw Data"] --> B["Preprocessing"]
    B --> B1["Categorical Encoding"]
    B --> B2["Outlier Removal"]
    B --> B3["Normalization"]
    B --> C["Train/Validation Split"]
    C --> D["CatBoost Training"]
    D --> D1["Hyperparameter Tuning"]
    D --> D2["Early Stopping"]
    D --> E["Validation Evaluation (ROC-AUC)"]
    E --> F["Final Model Predictions on Test Set"]

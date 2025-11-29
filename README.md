# Predicting-Loan-Payback---Kaggle
Building a loan payback prediction model for a kaggle competition using CAT boost classifier

## Project Overview

This project predicts the **probability of a loan being paid back** using historical loan data. The goal is to provide a **probabilistic risk score** for each loan applicant, allowing financial institutions to make informed lending decisions.

The model is evaluated using **ROC-AUC**, which measures how well the model ranks high-risk vs low-risk loans.

**Initial Model ROC-AUC:** 0.79347  
**Final Model ROC-AUC:** 0.92292  

---

## Dataset

The dataset contains both **numerical** and **categorical features** describing applicants and loans. Key features include:

- `gender`, `marital_status`, `education_level`, `employment_status`  
- `loan_purpose`, `grade_subgrade`  
- Loan amount, term, and other numeric variables  
- Target column: `loan_paid_back` (1 = loan paid back, 0 = default)  
- `id` column for unique identification  

---

## Data Preprocessing

### 1. Categorical Encoding
- Converted categorical columns to pandas **category type**.
- Applied **label encoding** so each category has a unique numeric code.
- Ensured that both **training and test sets** used consistent encoding.

### 2. Outlier Removal
- Used **Isolation Forest** with `contamination=0.1` to detect outliers in the training set.
- Removed outliers to **stabilize numeric features** and improve model performance.
- Example: After outlier removal, training rows reduced from `X_train.shape` to `X_train_clean.shape`.

### 3. Feature Normalization
- Used **MinMaxScaler** to scale numeric features to a [0,1] range.
- Applied scaling **after outlier removal** to prevent extreme values from distorting the normalization.
- Ensured validation and test sets were scaled using the **same scaler** fitted on the training data.

---

## Train-Validation Split

- Split the cleaned training data into **80% training** and **20% validation**.
- Used `stratify=y` to ensure **balanced target distribution** in both sets.
- Variables:
  - `X_train_clean` / `y_train_clean` → training set after outlier removal
  - `X_val_scaled` / `y_val` → validation set

---

## Model: CatBoostClassifier

The final model is a **gradient boosting classifier** using CatBoost with the following configuration:

```python
model = CatBoostClassifier(
    iterations=3500,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3,
    max_leaves=27,
    grow_policy="Lossguide",
    loss_function="Logloss",
    eval_metric="AUC",
    bootstrap_type="Bernoulli",
    subsample=0.8,
    random_seed=43,
    verbose=100,
    use_best_model=True,
    early_stopping_rounds=200
)

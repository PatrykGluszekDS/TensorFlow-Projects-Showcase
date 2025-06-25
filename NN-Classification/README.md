# 💳 Credit-Card Fraud Detector
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A production-ready deep-learning pipeline that spots fraudulent card transactions
among **284 807** European records with only **0.172 %** positives.

## 📦 Dataset

| Source | Date Range | Records | Fraud Ratio |
|--------|-----------|---------|-------------|
| [Credit Card Fraud Detection – ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 2 days in Sept 2013 | 284 807 | **0.172 %**

> The dataset contains VISA transactions by European cardholders. Features `V1–V28` are PCA-anonymised; `Time` is seconds elapsed since first transaction; `Amount` is the transaction € value; `Class` is the fraud label (`1` = fraud).


## 🔎 Exploratory Data Analysis

| Question | Insight |
|----------|---------|
| **How unbalanced is the target?** | Fraud accounts for **0.172 %** of 284 807 records. |
| **Do legitimate & fraudulent transactions differ in € Amount?** | Fraudsters slightly prefer mid-range amounts but overlap is large → Amount alone can’t separate classes |
| **Correlation structure?** | Off-diagonal cells for V1-V28 are ~0 (grey). That’s expected—these are PCA components designed to be linearly uncorrelated. Amount shows weak pos/neg links with a few components (e.g. V7, V20) |
| **Any time-based drift?** | Plot suggests that frauds could occur in bursts and could be coordinated attacks. |

<details>
<summary><strong>Key visuals produced</strong></summary>

* Class-imbalance bar chart  
* Log-scaled distribution of `Amount` for each class  
* Pearson-correlation heat-map (`V1–V28`, `Amount`)  
* Fraud-rate by 1-hour time bins
</details>


## 🛠️ Data Pre-processing

| Step | What & Why |
|------|------------|
| **1. Drop `Time`** | `Time` is just seconds-since-first-txn, not an intrinsic feature. It was removed to avoid data-leakage across time splits. |
| **2. Scale `Amount` ⇒ (0 – 1)** | Used `sklearn.preprocessing.MinMaxScaler`. Normalising helps neural nets converge faster. |
| **3. Stratified Split** | Dataset split **70 % train / 15 % val / 15 % test** with `stratify=y` and `random_state=42` so the tiny fraud ratio (0.172 %) is preserved in each subset. |
| **4. Artifact persistence** | Saved `X_train.npy`, `y_train.npy` |

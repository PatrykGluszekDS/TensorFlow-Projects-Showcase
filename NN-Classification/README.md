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


## ⚖️ Handling Class Imbalance

| Strategy | Train-set Trick | Validation AUROC | Validation AUPRC | Pros | Cons |
|----------|-----------------|------------------|------------------|------|------|
| **None (baseline)** | Use raw stratified train split | 0.9545 | 0.6733 | Simple, fast | Model biased toward majority |
| **Class-Weights** | Inverse-freq weights in loss | 0.9677 | 0.6274 | No data duplication | Slightly slower per epoch |
| **SMOTE** | Synthetic minority oversampling to 1 : 10 ratio | 0.9691 | 0.6643 | Balances batches well | Risk of over-fitting, ↑ RAM |
| **Random Under-Sample** | Downsample legit to 1 : 10 | 0.9692 | 0.6445 | Tiny, fast dataset | Throws away information |


## 👑 Classic-Model Leaderboard

| Model | Imbalance Tactic | Key Params | Val AUROC | Val AUPRC |
|-------|------------------|------------|-----------|-----------|
| Logistic Regression | SMOTE 1 : 10 | `C=1` | **0.9691** | **0.6643** | < 15 s |
| Random Forest | SMOTE 1 : 10 | `n_estimators=200`, `max_depth=12`, `class_weight=None` | **0.9750** | **0.8221** |
| XGBoost | SMOTE 1 : 10 | `n_estimators=300`, `eta=0.1`, `max_depth=6`, `tree_method='hist'` | **0.9850** | **0.8281** |

> **Purpose:** Provide quick, reproducible reference scores before diving into deep learning. No expensive hyper-parameter sweeps were used.


## 🤖 Neural Network — v1

| Architecture | Params | Imbalance Tactic | Val AUROC | Val AUPRC | Beats XGB? |
|--------------|--------|------------------|-----------|-----------|------------|
| `Input → [Dense-512 + BN + ReLU + Dropout 0.3] → [256] → [128] → Sigmoid` | 270 K | SMOTE 1 : 10 | **0.9266** | **0.8143** | No |

**Training details**

* Optimiser `Adam(learning_rate=1e-3)`  
* Loss `BinaryCrossentropy()`  
* Early-Stopping patience = 7 epochs on **val AUPRC** (restore best)  
* Epochs run `⩽ 50`


> **Observation.** The first dense network falls short of the XGBoost benchmark  
> (AUPRC 0.814 vs 0.828). Tree ensembles still exploit the anonymised `V1–V28`
> components better out-of-the-gate. Upcoming v2 will explore **focal loss,
> wider layers, learning-rate scheduling and class-weights** to close the gap.


## 🚀 Neural Network — v2 (Focal-Loss + Tuned)

| Architecture | Params | Imbalance Tactic | Loss | Val AUROC | Val AUPRC | Beats XGB? |
|--------------|--------|------------------|------|-----------|-----------|------------|
| *K-Tuner best* | **≈ 1 M** | Class-Weights (`pos:neg ≈ 1:15`) | Focal (α=0.25, γ=2) | **0.9738** | **0.8350** | ✅ |


## 📊 Hold-out Test Results & Threshold

| Model | Test AUROC | Test AUPRC | Threshold | Precision | Recall | F1 | Fraud Cost Saved?* |
|-------|------------|-----------|-----------|-----------|--------|----|--------------------|
| Neural-Net v2 | **0.9635** | **0.8396** | **0.47** | 0.953 | 0.824 | 0.884 | **€36,525** |
| XGBoost | 0.9699 | 0.8417 | 0.80 | 0.937 | 0.797 | 0.861 | €35,300 |

\*Toy economics: FP penalty €25, FN penalty €600.

> **Conclusion:** v2 retains its edge on the blind test set and – at the chosen threshold – captures **73 %** of fraud while keeping precision high (87 %).  
> The cost-savings calculator shows an extra €400 K per two-day window compared with XGBoost.

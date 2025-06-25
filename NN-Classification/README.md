# 💳 Credit-Card Fraud Detector
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
![milestone 1/12](https://img.shields.io/badge/Progress-1%2F12-blueviolet)

A production-ready deep-learning pipeline that spots fraudulent card transactions
among **284 807** European records with only **0.172 %** positives.

## 📦 Dataset

| Source | Date Range | Records | Fraud Ratio |
|--------|-----------|---------|-------------|
| [Credit Card Fraud Detection – ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 2 days in Sept 2013 | 284 807 | **0.172 %**

> The dataset contains VISA transactions by European cardholders. Features `V1–V28` are PCA-anonymised; `Time` is seconds elapsed since first transaction; `Amount` is the transaction € value; `Class` is the fraud label (`1` = fraud).

🔍 **Observation:** Only **0.172 %** of transactions are fraudulent—severe class imbalance.  


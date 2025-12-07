# Bike Sharing Daily Demand Forecasting with TensorFlow (Daily Aggregation)

## Project Overview

This project applies **time series forecasting** techniques using **TensorFlow/Keras** to predict **daily bike rental demand** in a city bike-sharing system.

The main goals are:

* Build an **end-to-end workflow** for time series forecasting on a real-world dataset.
* Start from a **simple univariate setup** (only daily rental counts) to solidify the fundamentals.
* Compare **naïve baselines** with **deep learning models** (Dense NN, LSTM/GRU, 1D CNN).
* Present results in a way that is understandable for **business stakeholders** and **technical staff**.

---

## Dataset

* **Source:** Kaggle – Bike Sharing Dataset (e.g. `lakshmi25npathi/bike-sharing-dataset`, file `day.csv`).
* **Date range used:** From **2011-01-01** to **2012-12-31**.
* **Number of observations:** 731 daily records (two full years, including leap year 2012).
* **Granularity used here:** **Daily** total rental counts.
* **Target variable:** `cnt` – total number of bikes rented on a given **day**.

In the first version of the project, I focus on a **univariate time series**:

> `y_t = total number of rentals on day t`

Original raw CSV is **not** committed to this repository. To reproduce the results, download the dataset from the original source or download via `kagglehub` in a notebook and point to `day.csv`.

---

## Problem Formulation

> Given the historical sequence of **daily total bike rentals**, forecast **future daily demand**.

This setup reflects a realistic question a bike sharing company might ask:

> "How many rentals do we expect tomorrow, based on past demand patterns?"

---

## Modeling Approach

### 1. Exploratory Data Analysis (EDA)

* Visualizing the daily rental series over time.
* Check for trends, seasonality (e.g. yearly, weekly patterns), and anomalies.

### 2. Baseline Models

Before using deep learning, simple baselines are implemented for reference:

1. **Naïve Forecast**
   `ŷ(t+1) = y(t)` (tomorrow = today).

### 3. Data Preparation for TensorFlow

* Implementing a **windowing function** to create input–label pairs for supervised learning. For example:

  * Input window: last 30 days of demand.
  * Label: demand on the next day.
* Using `tf.data.Dataset` pipelines for efficient batching and shuffling.

### 4. TensorFlow Models & Planned Experiments

Several neural network architectures are trained and compared.

#### 4.1 Baseline Model

0. **Naïve model (baseline)**

   * Forecasts the next value as the last observed one, e.g. `ŷ(t+1) = y(t)`.
   * Serves as a simple benchmark that all other models should beat.

#### 4.2 Core Dense Models (Univariate)

1. **Dense model, horizon = 1, window = 7**

   * Inputs: last 7 days of `daily_total_rentals`.
   * Output: next-day demand.

2. **Dense model, horizon = 1, window = 30**

   * Inputs: last 30 days.
   * Output: next-day demand.

3. **Dense model, horizon = 7, window = 30**

   * Inputs: last 30 days.
   * Output: vector of the next 7 days (multi-step forecast).

#### 4.3 Sequence and Convolutional Models

4. **Conv1D model**

   * 1D convolution over the input window (e.g. 30 days) to capture local temporal patterns.

5. **LSTM model**

   * Recurrent neural network treating the 7- or 30-day windows as sequences.

6. **Dense model with multivariate inputs**

   * Same idea as model 1 but augmented with weather features.

#### 4.4 Advanced & Ensemble Approaches

7. **N-BEATS-style model** (advanced experiment, if time allows).

8. **Ensemble model**

   * Combine predictions from several strong individual models (e.g. Dense + LSTM + Conv1D).

9. **Future prediction model / deployment-ready model**

   * A final, best-performing model prepared for generating forecasts beyond the dataset (e.g. next 7/30 days) and potentially for deployment.

All models are trained with:

* **Loss:** MAE or MSE.
* **Optimizer:** Adam.

### 5. Evaluation & Interpretation

* Comparing baselines vs neural models on the **test set**.
* Quantify improvement over naïve baseline.

The focus is not only on minimizing error but also on **explaining** what the models learn, when they perform well, and where they struggle (e.g. special events, holidays, unusual weather).

---

## ✅ Current Progress

* ✅ Loaded the **daily bike sharing data** from `day.csv` using a parsed datetime index (`dteday`).
* ✅ Aggregated to a univariate series `daily_total_rentals` and inspected the first values.
* ✅ Verified the dataset contains **731 daily observations** from 2011-01-01 to 2012-12-31 with no missing dates.
* ✅ Created an initial **time series line plot** of daily total rentals, showing clear yearly seasonality and overall growth in demand.
* ✅ Split the series into **chronological train/test sets** (80% / 20%):

  * Train: first 584 days.
  * Test: remaining 147 days.
* ✅ Visualized the split using a scatter plot of train vs. test points over time to confirm there is no leakage.
* ✅ Implemented a reusable helper function `plot_time_series(...)` to create consistent time series plots throughout the project.
* ✅ Implemented an `evaluate_preds(...)` utility that computes **MAE, MSE, RMSE, MAPE and MASE** (with MASE scaled by a non-seasonal naive benchmark).
* ✅ Trained and evaluated **Model 0 – Naïve baseline (horizon = 1)** on the test set.
* ✅ Implemented several neural baselines for **horizon = 1** (one-day-ahead forecast):

  * **Model 1 – Dense (window = 7)**.
  * **Model 2 – Dense (window = 30)**.
  * **Model 4 – Conv1D (window = 7)** using a 1D convolution over the last 7 days.
* ✅ Implemented **NumPy-based windowing utilities** (`make_windows`, `get_labelled_windows`, `make_train_test_splits`) to create supervised learning samples for different window sizes.
* ✅ For **Model 1 – Dense NN (window = 7, horizon = 1)**:

  * Created 724 windows of shape `(7,)` with labels `(1,)`, split into **579 train windows** and **145 test windows**.
  * Architecture: Dense(128, ReLU) → Dense(1, linear).
  * Trained with Adam on MAE loss, using a model checkpoint callback.
  * Evaluated and plotted predictions vs. test data.
* ✅ For **Model 2 – Dense NN (window = 30, horizon = 1)**:

  * Created 701 windows of shape `(30,)` with labels `(1,)`, split into **560 train windows** and **141 test windows**.
  * Architecture: Dense(128, ReLU) → Dense(1, linear) (same as Model 1, but with a longer input history).
  * Trained with Adam on MAE loss, again using a checkpoint callback.
  * Generated forecasts on the test windows, evaluated them with `evaluate_preds(...)`, and visualized predictions vs. test data.

Next immediate steps:

* Implement additional architectures (e.g. **Conv1D** and **LSTM** models) and compare against the Dense baselines.
* Further refine **multi-step forecasting** and compare different strategies (direct vs. recursive forecasts).

---

## 📊 Metrics & Results

### Results (Horizon = 1, Daily Forecast)

The first experiments focus on **one-day-ahead forecasts** using only the previous daily rental counts.

Approximate metrics on the test period:

| Model                                  | MAE (rentals) | RMSE (rentals) | MAPE (%) | MASE  |
| -------------------------------------- | ------------- | -------------- | -------- | ----- |
| Naïve (ŷₜ = yₜ₋₁)                      | ~883          | ~1287          | ~158     | ~1.00 |
| Dense NN (window = 7, horizon = 1)     | ~794          | ~1246          | ~217     | ~0.89 |
| Dense NN (window = 30, horizon = 1)    | ~843          | ~1256          | ~213     | ~0.94 |
| Conv1D + GAP (window = 7, horizon = 1) | ~931          | ~1423          | ~262     | ~1.04 |
| LSTM (window = 7, horizon = 1)         | ~855          | ~1309          | ~243     | ~0.96 |
| LSTM (64 units, tanh, unscaled input)  | ~5870         | ~6163          | ~100     | ~6.59 |

* **Model 1 vs. Naïve:** The 7-day Dense model reduces MAE and MASE by roughly **10%** compared to the naive baseline (MASE drops from ≈1.00 to ≈0.89), indicating that it leverages information from the past week to produce more accurate forecasts.
* **Model 2 vs. Model 1:** Extending the input window to 30 days does **not** further improve performance on this dataset; MAE and MASE slightly worsen (though still better than the naive model). This suggests that, for univariate daily counts, most of the predictive power is contained in the most recent week rather than a full month.
* **Model 4 vs. Dense models:** The Conv1D model with a 7-day window plus **GlobalAveragePooling1D** underperforms the simpler Dense baselines (higher MAE/MASE than Dense-7 and even slightly worse than the naive model). This suggests that, with such a short window and only one univariate feature, aggressively pooling over time destroys useful information about recency that the Dense models can exploit.
* **Model 5 (LSTM):** The LSTM-based sequence model outperforms the naive baseline (MASE < 1) but remains slightly worse than the best Dense-7 model. With only 7 timesteps and a single feature, the recurrent architecture does not bring a clear advantage over a well-tuned feed-forward network, but it demonstrates how to apply RNNs to time series.
* **Model 5.1 (LSTM with tanh + 64 units):** Switching to a smaller LSTM with the default `tanh` activation **without scaling the inputs** leads to a dramatic degradation in performance (MASE ≫ 1). Since daily rental counts are in the range of thousands, feeding them directly into `tanh`-gated recurrent units causes saturation and unstable gradients. This experiment highlights the importance of **feature scaling** (e.g. normalising series to [−1, 1]) when using tanh-based RNNs. For the final comparison, Model 5 is kept as the representative LSTM variant and Model 5.1 is treated as a negative result/ablation.
* **Error characteristics:** All one-day-ahead neural models tend to **smooth out day-to-day variability**, capturing the overall trend but sometimes underestimating sharp peaks or drops. The Conv1D + GAP model is even smoother, which explains its weaker performance.
* **MAPE caution:** The MAPE values are relatively high for all models due to days with very low rental counts (division by small actual values). Therefore, MAE, RMSE, and especially MASE are treated as the primary metrics for model comparison.

### Results (Horizon = 7, Weekly Forecast)

Model 3 extends the task to **multi-step forecasting**, predicting the next 7 days of demand from the previous 30 days. Metrics are averaged across all forecast horizons.

| Model                               | MAE (rentals) | RMSE (rentals) | MAPE (%) | MASE  |
| ----------------------------------- | ------------- | -------------- | -------- | ----- |
| Dense NN (window = 30, horizon = 7) | ~1036         | ~1577          | ~259     | ~1.16 |

* **Difficulty of multi-step forecasting:** As expected, predicting an entire week ahead is harder than predicting just the next day. The multi-step Dense model has larger errors and **MASE > 1**, meaning that, on average, it does not outperform a simple non-seasonal naive benchmark.
* **Behaviour:** The 7-day forecasts are smooth and follow the general trend but struggle with sharp local variations, especially further into the forecast horizon.
* **Next directions:** This motivates trying more sequence-aware architectures such as **Conv1D** or **LSTM/GRU** networks, and potentially incorporating calendar/weather covariates for richer multi-step models.

As more models are trained (ensembles, multivariate models, etc.), additional rows will be added to these tables along with short commentary on their relative performance.

Model 3 extends the task to **multi-step forecasting**, predicting the next 7 days of demand from the previous 30 days. Metrics are averaged across all forecast horizons.

| Model                               | MAE (rentals) | RMSE (rentals) | MAPE (%) | MASE  |
| ----------------------------------- | ------------- | -------------- | -------- | ----- |
| Dense NN (window = 30, horizon = 7) | ~1036         | ~1577          | ~259     | ~1.16 |

* **Difficulty of multi-step forecasting:** As expected, predicting an entire week ahead is harder than predicting just the next day. The multi-step Dense model has larger errors and **MASE > 1**, meaning that, on average, it does not outperform a simple non-seasonal naive benchmark.
* **Behaviour:** The 7-day forecasts are smooth and follow the general trend but struggle with sharp local variations, especially further into the forecast horizon.
* **Next directions:** This motivates trying more sequence-aware architectures such as **Conv1D** or **LSTM/GRU** networks, and potentially incorporating calendar/weather covariates for richer multi-step models.

As more models are trained (LSTM, ensembles, etc.), additional rows will be added to these tables along with short commentary on their relative performance.

Model 3 extends the task to **multi-step forecasting**, predicting the next 7 days of demand from the previous 30 days. Metrics are averaged across all forecast horizons.

| Model                               | MAE (rentals) | RMSE (rentals) | MAPE (%) | MASE  |
| ----------------------------------- | ------------- | -------------- | -------- | ----- |
| Dense NN (window = 30, horizon = 7) | ~1036         | ~1577          | ~259     | ~1.16 |

* **Difficulty of multi-step forecasting:** As expected, predicting an entire week ahead is harder than predicting just the next day. The multi-step Dense model has larger errors and **MASE > 1**, meaning that, on average, it does not outperform a simple non-seasonal naive benchmark.
* **Behaviour:** The 7-day forecasts are smooth and follow the general trend but struggle with sharp local variations, especially further into the forecast horizon.
* **Next directions:** This motivates trying more sequence-aware architectures such as **Conv1D** or **LSTM/GRU** networks, and potentially incorporating calendar/weather covariates for richer multi-step models.

As more models are trained (Conv1D, LSTM, ensembles, etc.), additional rows will be added to these tables along with short commentary on their relative performance.

---

## 🔮 Roadmap / Possible Extensions

Planned or potential future improvements:

* 🔁 **Multi-step forecasting** (predict demand for the next 7 or 14 days).
* 🎛️ Incorporate **calendar features** (weekday, weekend/weekday, holidays).
* 🌦️ Add **weather and season** variables for a multivariate model.
* 🔍 Hyperparameter tuning (e.g. with KerasTuner or Optuna).
* 📈 Try more advanced architectures (e.g. Temporal Convolutional Networks, N-BEATS-style models).
* 🧪 Model deployment demo (simple API or Streamlit dashboard for prediction).

---

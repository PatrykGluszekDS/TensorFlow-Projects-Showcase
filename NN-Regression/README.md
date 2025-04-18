# 🚗 Vehicle Fuel Efficiency Prediction (MPG) using Neural Networks and TensorFlow

This project aims to predict vehicle fuel efficiency (measured in miles per gallon - MPG) based on various vehicle characteristics, using neural network regression built in TensorFlow.

## 🎯 Project Goals
- Accurately predict fuel efficiency (MPG) from given vehicle features.
- Demonstrate regression modeling skills using TensorFlow and Keras.
- Clearly interpret and visualize the performance of the neural network.

## 📂 Dataset
- **Dataset Name:** Auto MPG Dataset
- **Source:** [UCI Machine Learning Repository - Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
- **Features:**
  - Cylinders, displacement, horsepower, weight, acceleration, model year, origin
- **Target Variable:**
  - MPG (Miles Per Gallon)

## 🛠️ Technologies Used
- Python
- TensorFlow & Keras API
- Scikit-learn
- NumPy & Pandas
- Matplotlib & Seaborn
- Jupyter Notebook
  
## 🚩 Methodology
1. **Data Acquisition and Exploration** [x]
    - Data loading, checking for missing values, visual exploration.
  
2. **Data Preprocessing** [x]
    - Handling missing values, encoding categorical features, scaling data.

3. **Model Development** [x]
    - Building and training a Neural Network Regression model.

4. **Evaluation & Tuning** [x]
    - Performance metrics: MAE, MSE, RMSE, R².
    - Hyperparameter tuning: Layers, neurons, activation functions.

5. **Visualization & Interpretation** [x]
    - Actual vs. predicted MPG plots, error analysis.

## 🧠 Models Developed

### ✅ Model 1: Baseline Model
- **Architecture**: 1 hidden layer with 32 neurons (ReLU)
- **Epochs**: 10
- **Batch Size**: 16

| Metric | Value |
|--------|-------|
| MAE    | 15.92 |
| MSE    | 328.56 |
| RMSE   | 18.13 |
| R²     | -5.11 |

---

### 🔧 Model 2: Tuning Hidden Layers + Neurons
- **Architecture**: 2 hidden layers (64 and 32 neurons)
- **Epochs**: 10
- **Batch Size**: 16

| Metric | Value |
|--------|-------|
| MAE    | 4.58 |
| MSE    | 29.86 |
| RMSE   | 5.46 |
| R²     | 0.445 |

---

### 🏁 Model 3: Tuned Epochs (Improved Training)
- **Architecture**: Same as Model 2
- **Epochs**: 100
- **Batch Size**: 16

| Metric | Value |
|--------|-------|
| MAE    | 1.71 |
| MSE    | 5.10 |
| RMSE   | 2.26 |
| R²     | 0.91 |

---

## 📊 Model Comparison

| Model    | MAE   | MSE     | RMSE   | R²     |
|----------|--------|----------|--------|--------|
| Model 1  | 15.92  | 328.56   | 18.13  | -5.11  |
| Model 2  | 4.58   | 29.86    | 5.46   | 0.445  |
| Model 3  | 1.71   | 5.10     | 2.26   | 0.91   |

---

## 📚 Comparison with Scientific Literature

This project replicates and improves on the work published in the following paper:

> **"Artificial Neural Network Model for Prediction of Fuel Consumption of a Car"**  
> *International Journal of Artificial Intelligence and Scientific Research, 2020*  
> [Read paper (PDF)](http://ijeais.org/wp-content/uploads/2020/11/IJAISR201102.pdf)

| Feature                     | My Project                                 | IJAISR Paper (2020)                    |
|----------------------------|--------------------------------------------|----------------------------------------|
| **Dataset**                | UCI Auto MPG                                | UCI Auto MPG                           |
| **Framework**              | TensorFlow/Keras (Python)                  | JustNN GUI Tool                        |
| **Architecture**           | Tested multiple: 1H-32, 2H-64-32            | Fixed: 2H, 3 total neurons             |
| **Training Epochs**        | Tuned manually: 10–100                      | 7683 epochs                            |
| **Metrics Used**           | MAE, RMSE, MSE, R²                          | Classification Accuracy (not ideal)   |
| **Best R²**                | **0.91**                                    | Not reported                           |
| **Hyperparameter Tuning**  | Manual tuning in code                      | Trial and error via GUI                |
| **Reproducibility**        | Full code & notebooks available            | No code provided                       |

### 🧠 Insight

This project not only reproduced the paper's experiment using modern frameworks but also demonstrated **better results**, **clearer methodology**, and **stronger metrics** — all with **fewer training epochs**.

---

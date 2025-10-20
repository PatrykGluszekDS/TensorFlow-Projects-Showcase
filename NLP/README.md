# News Topic Classification (AG News)

**Goal:** classify short news headlines/descriptions into four topics: **Business**, **Sci/Tech**, **Sports**, **World**.

**Highlights**

* **EDA** (balanced labels; typical length ≈ 30–50 tokens).
* Strong classical baseline: **TF‑IDF + Multinomial NB** → **90.86%** val acc.
* Best deep model (val): **Hybrid** = *USE sentence embeddings* + *character branch* → **92.05%** val acc.
* **Locked‑in test** (so far): **Model 6 (USE feature extractor)** → **91.07%** acc • **0.911** F1.

---

## 1) Dataset

* **AG News** (4 classes), split into **train/val/test** (validation split created from train).
* Class balance is ~uniform across splits.
* Length stats (train):

  * Tokens p50 / p75 / p95: **37 / 43 / 53**
  * Characters p50 / p75 / p95: **232 / 266 / 343**
* From EDA histograms: long tail up to ~170 tokens, but most samples are short.

> **Decision:** set `TextVectorization(output_sequence_length=53)` to cover ~95% of samples and keep models efficient. Character branch uses `output_seq_char_len ≈ 343` (p95).

---

## 2) Preprocessing

**Token level**

* `TextVectorization(standardize="lower_and_strip_punctuation", split="whitespace", output_sequence_length=53)`
* **Embedding:** `Embedding(vocab_size, 128, mask_zero=True)`

**Character level**

* Alphabet: `a–z`, digits, punctuation (incl. space + OOV).
* `TextVectorization(output_mode="int", output_sequence_length=343)`
* **Embedding:** `Embedding(len(char_vocab), 25, mask_zero=True)`

**Labels**

* Encoded with `LabelEncoder` and one‑hot for Keras training.

---

## 3) Models

**Baseline**

* **Model 0**: TF‑IDF → **MultinomialNB**

**Token embeddings (learned)**

* **Model 1**: Simple Dense (Token Embedding → GlobalAvgPool → Dense)
* **Model 2**: LSTM (64)
* **Model 3**: GRU (64)
* **Model 4**: Bidirectional LSTM (64)
* **Model 5**: Conv1D (filters=64, k=5) + GlobalAvgPool

**Pretrained sentence embeddings**

* **Model 6**: **USE (Universal Sentence Encoder, TF‑Hub)** as **feature extractor** (frozen) → Dense

**Character models**

* **Model 7**: Conv1D over char embeddings

**Hybrid**

* **Model 8**: **USE branch** (Dense 128 ReLU) **+** **char branch** (Char Embedding → BiLSTM(24)) → Concatenate → Dropout → Dense 128 → Dropout → Softmax

**Training setup**

* Optimizer: `Adam`
* Loss: `categorical_crossentropy`
* Metrics: `accuracy`
* **Callbacks:** `EarlyStopping(monitor="val_loss" or "val_accuracy", patience=3–4, restore_best_weights=True)` and `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6)`
* Batch size: 32

---

## 4) Validation Results

| model                          |    accuracy | precision |   recall |       f1 |
| ------------------------------ | ----------: | --------: | -------: | -------: |
| baseline (TF‑IDF + MNB)        | **90.8583** |  0.908264 | 0.908583 | 0.908331 |
| model_1_simple_dense_model     | **91.1667** |  0.911485 | 0.911667 | 0.911503 |
| model_2_lstm                   |     89.8167 |  0.898415 | 0.898167 | 0.898051 |
| model_3_gru                    |     89.6750 |  0.897551 | 0.896750 | 0.896607 |
| model_4_bidirectional_rnn      |     89.8667 |  0.898565 | 0.898667 | 0.898438 |
| model_5_conv1d_token_embed     |     89.7500 |  0.897233 | 0.897500 | 0.897292 |
| **model_6_feature_extraction** | **91.6083** |  0.916617 | 0.916083 | 0.915980 |
| model_7_conv1d_char_embed      |     83.0083 |  0.832167 | 0.830083 | 0.828944 |
| **model_8_hybrid_embed**       | **92.0500** |  0.920761 | 0.920500 | 0.920395 |

**Observations**

* Classical baseline is **strong** (90.86%).
* **USE** (Model 6) beats token‑only deep models.
* **Hybrid (Model 8)** edges out Model 6 on validation.
* **Char‑only** lags (as expected for clean English news).

---

## 5) Test‑Set Results (Locked‑in)

Evaluate the top models on the unseen **test** split and record metrics.

* **Model 6 – USE feature extractor (frozen)**

  * **Accuracy:** **91.0658%**
  * **Precision:** 0.9111 • **Recall:** 0.9107 • **Macro‑F1:** **0.9106**
  * (`model_6_test_results` from notebook)

---

## 6) Reproduce

### Train & Evaluate (notebook flow)

1. **EDA**: run cells to visualize label counts and length histograms.
2. **Preprocessing**: build token/char vectorizers and embeddings; fit on train only.
3. **Baselines & Models**: train Models 0–8 with callbacks.
4. **Validation leaderboard**: generate the comparison table.
5. **Lock in test results**:

   * For USE/Hybrid:

     ```python
     # Example for Model 6 (already done)
     test_probs = model_6.predict(test_dataset)
     test_preds = test_probs.argmax(axis=1)
     # calculate_results(y_true=test_labels_encoded, y_pred=test_preds)
     ```
   * For Model 8 (hybrid):

     ```python
     test_probs = model_8.predict(test_char_token_dataset)
     test_preds = test_probs.argmax(axis=1)
     ```
6. **Artifacts**: save metrics JSON, classification report, confusion matrix, and the Keras SavedModel.

---

## 8) What worked / What didn’t

**Worked**

* Pretrained sentence embeddings (USE) provide the largest jump over token‑only models.
* Hybridizing **USE + character** features adds a small but consistent gain on val.
* EarlyStopping + ReduceLROnPlateau prevents overfitting on short sequences.

**Didn’t help much**

* Pure RNNs (LSTM/GRU) over short sequences; Conv1D similar story.
* Char‑only model underperforms on this clean dataset.

---

## 9) Next Steps

* Evaluate **Model 8** on the **test** set and drop the confusion matrix + metrics.
* Try **fine‑tuning** USE (small LR, few epochs) and compare vs feature extraction.
* Swap USE for a modern small encoder.

---

### Short description

*Built a news topic classifier (AG News) with full EDA, strong classical baseline (TF‑IDF + Multinomial NB), and multiple deep architectures. Best validation model (Hybrid: USE + Char‑BiLSTM) reached **92.05%** accuracy; **locked‑in test** for USE feature extractor: **91.07%** accuracy / **0.911 macro‑F1**. Packaged reproducible notebook*

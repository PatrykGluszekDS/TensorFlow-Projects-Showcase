# AgriAid - Plant-Disease Diagnosis 🌿📸

**Plant Disease Diagnosis with Transfer Learning (EfficientNet, ResNet)**

> Identify 38 crop‑disease categories from 54 k+ leaf images, package the best model for on‑device inference.

## Project Highlights

* 🔍 **Transfer Learning**: EfficientNet, ResNet50
* 🧪 **Rigorous Experiments**: baseline, fine‑tuning, loss functions, data‑augmentation.
* 🌱 **Impact**: Helps farmers diagnose diseases early.

## Dataset

* **PlantVillage** (`plant_village` in TFDS) — 54 303 RGB images, 256 × 256 px, 38 classes (healthy + diseased leaves). ([tensorflow.org](https://www.tensorflow.org/datasets/catalog/plant_village))
* Predefined `train`/`validation`/`test` splits created inside the notebook for reproducibility.

## Exploratory Data Analysis

> **RAM‑friendly tip:** If Colab runs out of memory, skip the in‑memory stratified split and use TFDS percentage subsplits instead (e.g. `tfds.load('plant_village', split=['train[:80%]','train[80%:90%]','train[90%:]'], as_supervised=True)`). This streams images from disk and keeps peak RAM low.

* **Split sizes:** 43 442 (train) · 5 431 (val) · 5 430 (test) — checked with `tf.data.experimental.cardinality(...)`.

## Data Preprocessing

All three splits are piped through the same lightweight preprocessing function before entering the model:

```python
@tf.function
def preprocess_img(image, label, img_shape=224):
    """Resizes and casts the PlantVillage image to float32."""
    image = tf.image.resize(image, [img_shape, img_shape])          # 256×256 → 224×224
    # NOTE: EfficientNet/ResNet built‑in preprocessing includes scaling; additional /255 not required.
    return tf.cast(image, tf.float32), label

# Apply, shuffle, batch, prefetch
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

ds_train = (ds_train
            .map(preprocess_img, num_parallel_calls=AUTOTUNE)
            .shuffle(1000, seed=SEED)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

ds_val = (ds_val
          .map(preprocess_img, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

ds_test = (ds_test
           .map(preprocess_img, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))
```

**Why this matters**

* Resizing early keeps GPU utilisation high and avoids on‑device resize overhead.
* `.shuffle → .batch → .prefetch` ensures an efficient input pipeline that overlaps data loading with model execution.

## Getting Started

1. **Open in Colab** → `Runtime → Change runtime type → GPU`.
2. Run the notebook.

## Experiments & Results

| #  | Backbone & Variant                       | Training strategy               | Input px | Val Acc     | Test Acc    | Notes                                             |
| -- | ---------------------------------------- | ------------------------------- | -------- | ----------- | ----------- | ------------------------------------------------- |
| 1  | EfficientNetV2‑B0                        | frozen feature extractor        | 224      | **0.982**   | **0.984**   | Mixed‑precision; 5 epochs; ES + RLRP              |
| 2  | EfficientNetV2‑B0 + **Aug**              | frozen extractor + in‑model aug | 224      | 0.954       | 0.960       | Flip, rotate ±0.2, zoom/height/width 0.2          |
| 3  | EfficientNetV2‑B0 + **Aug + FT30**       | fine‑tune last 30 layers        | 224      | 0.968       | 0.971       | 35 epochs total; lr 1e‑4→plateau                  |
| 4  | ResNet50 (**wrong preprocessing**)       | frozen feature extractor        | 224      | 0.407       | 0.430       | Used only Rescaling 1/255 → mismatch              |
| 5  | **ResNet50 (official preprocess)**       | frozen feature extractor        | 224      | **0.982**   | **0.985**   | `resnet.preprocess_input` (BGR + means), 5 epochs |
| 6  | ResNet50 + **Aug** (official preprocess) | frozen extractor + in‑model aug | 224      | 0.946       | 0.949       | Flip/rotate/zoom/height/width 0.2                 |
| 7  | **ResNet50 + Aug + FT30**                | fine‑tune last 30 layers        | 224      | **≈ 0.989** | **≈ 0.990** | 35 epochs total; RLRP stepped LR; ES enabled      |

**Exp‑2 summary (EfficientNetV2‑B0 + Aug)**

* **Data augmentation**: `RandomFlip('horizontal')`, `RandomRotation(0.2)`, `RandomHeight(0.2)`, `RandomWidth(0.2)`, `RandomZoom(0.2)`.
* Backbone **frozen**, mixed‑precision enabled.
* Adam, lr = 1 e‑3; callbacks: EarlyStopping, ReduceLROnPlateau.
* **5 epochs** → **Val 0.954 / Test 0.960**, loss ≈ 0.13.

---

**Exp‑3 summary (EfficientNetV2‑B0 + Aug, fine‑tuned 30 layers)**

* Starting from Exp‑2 weights, **unfroze last 30 layers** of the backbone (`base_model_2.layers[-30:]`).
* Re‑compiled with Adam, lr = 4 e‑5.
* Trained to **epoch 25/35** with EarlyStopping + ReduceLROnPlateau.
* Final metrics: **Val ≈ 0.968**, **Test = 0.971**, loss ≈ 0.088.

---

**Exp‑4 summary (ResNet50 baseline – wrong preprocessing)**

* Feature‑extraction pass with **`tf.keras.applications.ResNet50`** (`include_top=False`, `weights='imagenet'`).
* Used only `layers.Rescaling(1./255)` → **no ImageNet mean subtraction / RGB↔BGR swap** → distribution mismatch.
* Mixed‑precision on; Adam lr = 1e‑3; 5 epochs.
* Results: **Val ≈ 0.407**, **Test ≈ 0.430**, loss ≈ 2.22.
* Conclusion: preprocessing mismatch severely hurts performance; fix with `resnet.preprocess_input` or switch to ResNet50V2 + \[-1,1] scaling.

---

**Exp‑5 summary (ResNet50 with official preprocessing)**

* Rebuilt baseline with **`from tensorflow.keras.applications.resnet import preprocess_input`** ahead of the backbone (does **BGR conversion** + **channel‑wise mean subtraction**).
* Backbone **frozen**; no augmentation; mixed‑precision; Adam lr = 1 e‑3; **5 epochs**.
* Final metrics: **Val ≈ 0.982**, **Test ≈ 0.986**, loss ≈ 0.045.
* Takeaway: correct preprocessing restores expected performance; now competitive with EfficientNetV2‑B0.

---

**Exp‑6 summary (ResNet50 + Aug, official preprocessing)**

* Same official preprocessing as Exp‑5; **in‑model augmentation**: `RandomFlip('horizontal')`, `RandomRotation(0.2)`, `RandomHeight(0.2)`, `RandomWidth(0.2)`, `RandomZoom(0.2)`.
* Backbone **frozen**; 5 epochs; mixed‑precision; Adam lr = 1e‑3.
* Final metrics: **Val ≈ 0.946**, **Test ≈ 0.949**, loss ≈ 0.160.
* Interpretation: strong aug with a frozen backbone reduces accuracy vs no‑aug baseline. Options: (1) lighten aug (rotation/zoom/height/width → 0.1), (2) train longer, or (3) **fine‑tune the last 30–40 layers** at lr ≈ 1e‑4 with the same aug.

**Exp‑7 summary (ResNet50 + augmentation, fine‑tuned last 30 layers)**

* Continued from Exp‑6: set `base_model_6.trainable=True`, then **re‑froze all but the last 30 layers**.
* Same official `resnet.preprocess_input`; kept in‑model augmentation.
* Optimizer Adam; **ReduceLROnPlateau** automatically reduced LR (2e‑4 → 4e‑5 → 8e‑6, etc.).
* Trained to epoch \~16/35 (EarlyStopping active). Final metrics: **Val ≈ 0.989**, **Test ≈ 0.990**, loss ≈ 0.034.
* Takeaway: fine‑tuning with moderate aug surpasses all previous runs; this is a strong candidate for the model ready to export.

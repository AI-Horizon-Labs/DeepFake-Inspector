# 📌 DeepFake-Inspector

**A robust and efficient DeepFake detection pipeline using fine-tuned EfficientNet-B0, progressive training, data augmentation, and synthetic image generation.**

## 🧠 Overview

This project proposes a robust DeepFake detection model combining **transfer learning**, **progressive fine-tuning**, and **threshold calibration** on public datasets. We used the EfficientNet-B0 architecture to balance **computational cost** and **accuracy**, with impressive results after 50+ hours of GPU training and evaluation on over **179,000 images**.

> **Final Accuracy:** 98%  
> **Best Threshold:** 0.3495  
> **Precision/Recall:** 0.98 / 0.98  
> **Framework:** TensorFlow + Keras  
> **Hardware:** NVIDIA RTX 5070 Ti (CUDA 12.2 + cuDNN 8.9.5)

---

## 📂 Datasets

We used the dataset:

- **deepfake and real images** ([Manjul Karki](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images))
- **OpenForensics** ([Le et al., 2021](https://zenodo.org/records/5528418#.YpdlS2hBzDd))

A progressive subset strategy was used:

| Subset  | Total Samples | Class Balance |
|---------|---------------|----------------|
| Mini    | ~1,068        | 50% real/fake  |
| Middle  | ~5,517        | ~54% real      |
| Big     | ~11,000       | ~50% real      |
| Full    | ~179,430       | ~50% real/fake  |

---

## ⚙️ Architecture

- **Backbone**: `EfficientNet-B0` (pretrained on ImageNet)
- **Custom classifier head**:
  - `GlobalAveragePooling2D`
  - Dense (512, ReLU, Dropout 0.4)
  - Dense (256, ReLU, Dropout 0.4)
  - Dense (1024, ReLU, Dropout 0.5)
  - Output: Sigmoid activation

---

## 🔁 Training Strategy

Three **fine-tuning stages** were performed:

| Stage     | Description                        | Accuracy | Recall Real/Fake | AUC   |
|-----------|------------------------------------|----------|------------------|-------|
| Stage 1   | Frozen base, dense layers only     | 50%      | 0 / 1            | 0.50  |
| Stage 2   | Unfrozen last 20–40 layers         | 55%      | 0.01 / 1         | <0.60 |
| Stage 3   | Full fine-tuning (50 epochs)       | 98%      | 0.98 / 0.99      | ~1.00 |

Hyperparameters (final run):

- Epochs: `50`
- Batch Size: `32`
- Optimizer: `Adam`
- LR Schedule: `ReduceLROnPlateau`
- EarlyStopping: `patience=10`

---

## 🧪 Preprocessing

- Image Resize: `224×224`
- Normalization: `[0, 1]`
- Data Augmentation:
  - Horizontal flip
  - Rotation ±10°
  - Zoom ±10%
- Synthetic Data Generation:
  - Using `OmniGen` for expression, lighting, and adversarial variants

---

## 📊 Results

### 📊 Results per Training Phase

| **Phase**          | **Threshold (τ)** | **Accuracy** | **Precision (Real/Fake)** | **Recall (Real/Fake)** | **Support** |
| ------------------ | ----------------- | ------------ | ------------------------- | ---------------------- | ----------- |
| **Frozen Base**    | –                 | ~0.50        | – / 0.55                  | 0.00 / 1.00            | –           |
| **Partial Tuning** | 1.068             | 0.55         | 0.80 / 0.55               | 0.01 / 1.00            | 1.103       |
| **Phase 1**        | 0.1591            | 0.80         | 0.73 / 0.90               | 0.93 / 0.67            | 2.181       |
| **Phase 2**        | 0.0731            | 0.77         | 0.71 / 0.88               | 0.91 / 0.64            | 7.885       |
| **Phase 3**        | 0.3495            | 0.98         | 0.98 / 0.98               | 0.98 / 0.99            | 179.430     |


### ✅ Confusion Matrix (Final Model, τ = 0.3495)

|          | Pred Real | Pred Fake |
|----------|-----------|-----------|
| **Real** | 87,881    | 1,789     |
| **Fake** | 1,794     | 87,966    |

### 📈 Metrics (Final Phase)

| Metric      | Value |
|-------------|-------|
| Accuracy    | 98%   |
| Precision   | 0.98  |
| Recall      | 0.98  |
| F1-Score    | 0.98  |
| AUC         | ~1.00 |
| Threshold   | 0.3495 (optimal via F1)

---

## 📁 Folder Structure

```
.
├── datasets/
│   ├── Complete/Train/ ##not provided
│   ├── Complete/Validation/
│   ├── mini-dataset/
├── models/
│   └── patience-5/models
│   └── patience-10/models
├── model.py
├── gpu.py
├── README.md
└── requirements.txt
```

---

## 💾 How to Save and Load the Model

```python
# Save
model.save("models/patience-{number}/model")

# Load
from tensorflow.keras.models import load_model
model = load_model("models/patience-{number}/model")
```

---

## ✅ Threshold Optimization

Threshold τ = 0.5 was suboptimal due to imbalanced probabilities. We used the **Precision–Recall curve** and maximized **F1-score** to choose the best threshold.

```python
from sklearn.metrics import precision_recall_curve

prec, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
f1_scores = np.nan_to_num(f1_scores)
best_threshold = thresholds[np.argmax(f1_scores)]
```

---

## ⚠️ Limitations & Ethics

- The model is **intra-dataset only**.
- Needs **cross-dataset** validation, video tests, and multiple seeds.
- Avoid misuse: DeepFake detection must protect privacy, not enforce surveillance.
- Models should be audited for **biases**, especially on underrepresented faces.

---

## 🙏 Acknowledgements

This research was supported by:

- CAPES – Financial Code 001
- FAPERGS – Grants 24/2551-0001368-7 and 24/2551-0000726-1

---

## 🔗 Citation

If you use this code or dataset setup, please cite the associated paper:

> **Combate à Falsificação Digital: Um Modelo de Detecção de Imagens DeepFake com Aprendizado Profundo**  
> Vinicius N. Lopes, Tadeu Jenuario, Vitor G. Balsanello, Diego Kreutz, Dionatan R. Schmidt, Eliezer Flores, Elder Rodrigues  
> UNIPAMPA - Programa de Pós-Graduação em Engenharia de Software (PPGES)

📎 [PDF here]()

## 🚀 How to Run (GPU - NVIDIA Docker Image)

This project is designed to run inside a container using the **NVIDIA TensorFlow Docker image**, fully utilizing GPU acceleration.

### 🛠️ Environment Setup

This project was developed and executed using GPU acceleration via Docker and the official NVIDIA TensorFlow image, making it highly reproducible across systems with compatible hardware.

#### 💻 System Specifications

| Component     | Description                               |
| ------------- | ----------------------------------------- |
| OS (Host)     | Windows 11                                |
| WSL Version   | Ubuntu 22.04 LTS WSL2                     |
| GPU           | NVIDIA GeForce RTX 5070 Ti                |
| Docker        | Docker Desktop with WSL2 Integration      |
| NVIDIA Driver | Version 535+ (CUDA 12.2 support)          |
| Docker Image  | `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` |
| TensorFlow    | 2.15+ (as included in the container)      |
| Python        | 3.10 (as included in the container)       |

#### 📦 Required Packages

These packages are required to run (some inside in NVIDIA container):

TensorFlow 2.15+

Keras (via TensorFlow)

NumPy

Matplotlib

Scikit-learn


#### 🧪 Environment Setup (Optional but Recommended)

To avoid dependency conflicts and ensure reproducibility, it is strongly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

After activating the virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

⚠️ Additional Dependencies

Depending on the state of the NVIDIA container, you may need to install additional system libraries manually.

### 1. Pull NVIDIA Docker Image (if not yet pulled)
```bash
docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```

### 2. Run Container with GPU and Mounted Volume

#### 🔐 NVIDIA NGC Authentication (Required)

This project uses an **official NVIDIA Docker image** hosted at `nvcr.io`. You must authenticate before pulling the image.

#### ✅ Authenticate with Docker

1. Go to: [https://ngc.nvidia.com/setup](https://ngc.nvidia.com/setup)
2. Generate your **API Key**
3. Run:

```bash
docker login nvcr.io
```

When prompted:
- **Username:** `$oauthtoken`
- **Password:** your **NGC API key**

⚠️ Heads-up

This authentication step is mandatory.
If you skip it, Docker will return a 401 Unauthorized or access denied when trying to pull the image.

#### 🚀 Running the Container

Assuming your project and dataset are inside `/c/Deepfake` on your host machine:

```bash
docker run --gpus all -it --rm   --shm-size=1g   --ulimit memlock=-1   --ulimit stack=67108864   -v /c/Deepfake:/workspace   nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```

### 3. Inside the Docker Container

```bash
cd /workspace (if necessary)
python model.py (to training)
python validate_model.py (to validate the validation dataset)
python validate_image.py (to detect if one image is Deepfake)
```

Replace `model.py` with your training script filename.

---

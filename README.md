# 🧠 DeepFake Detection with EfficientNet

This project focuses on detecting deepfakes using the EfficientNet-B0 architecture, trained on both real and synthetic images generated via OmniGen.

## 📌 Overview

We use a convolutional neural network based on **EfficientNet-B0** to classify images as **real** or **deepfake**. The goal is to assess the model's performance across different dataset sizes, using fine-tuning and evaluation metrics.

## 📂 Project Structure

- `datasets/`: real and synthetic (OmniGen) images (not available).
- `models/`: saved model checkpoints (not avaible).
- `results/`: classification reports, precision-recall curves, and plots (not available).

## 🧪 Techniques Used

- 🔸 **EfficientNet-B0** (initially frozen, then unfrozen)
- 🔸 Two-phase training:
  - Initial training with frozen base
  - Fine-tuning with partially and fully unfrozen layers
- 🔸 Optimizer: **Adam**
  - `1e-3` during pretraining
  - `1e-5` during fine-tuning
  - `1e-8` during fine-tuning
- 🔸 Custom layers:
  - `GlobalAveragePooling2D()`
  - `Dense(1024, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(512, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(256, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(1, activation='sigmoid')`
- 🔸 Data augmentation
- 🔸 Class weighting to handle class imbalance
- 🔸 Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `AUC`
- 🔸 Threshold optimization using `precision_recall_curve` for best F1-score

## 📊 Datasets Used

| Dataset     | Total Images | Notes                                         |
|-------------|--------------|-----------------------------------------------|
| Dataset 1   | 1,068        | Small, model underfit observed                |
| Dataset 2   | 5,517        | Medium-size, moderate performance             |
| Dataset 3   | 10,905       | Stable performance, recall significantly better |
| Dataset 4   | 179,430      | Large-scale dataset, training in progress     |

## ⚠️ Key Challenges

- Underfitting with small datasets.
- Very high recall for one class and near-zero for the other.
- AUC fluctuating between 0.49 and 0.57 on smaller datasets.
- Threshold tuning was necessary to balance class performance.

## 📌 Summary

EfficientNet-B0 proved to be suitable for experimental contexts and smaller datasets due to its lightweight structure and solid performance. The use of OmniGen-generated images (up to 3 inputs with expressions and different angles) added variability and improved the model's robustness.

## 🚀 How to Run

....

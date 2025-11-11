import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import json
import gc

MODEL_PATH = "/workspace/models/patience-10/deepfake_detector_model.keras"
VAL_DIR = "/workspace/datasets/Dataset/Validation"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 128 if tf.config.list_physical_devices('GPU') else 32
MAX_BATCHES = None  
SAVE_PARTS = True   
SAVE_PATH = os.path.join(os.path.dirname(MODEL_PATH), "best_threshold.json")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation dataset...")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = val_ds.class_names
print(f"Detected classes: {class_names}")

val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=tf.data.AUTOTUNE)
if MAX_BATCHES:
    val_ds = val_ds.take(MAX_BATCHES)
    print(f"Using only {MAX_BATCHES * BATCH_SIZE} images (quick sampling)")

print("\nGenerating predictions robustly...")
y_true, y_scores = [], []
part = 0

for i, (images, labels) in enumerate(val_ds):
    try:
        preds = model.predict(images, verbose=0).flatten()
        y_scores.extend(preds)
        y_true.extend(labels.numpy())

        if SAVE_PARTS and (i + 1) % 50 == 0:
            np.save(f"preds_part_{part}.npy", np.array(y_scores))
            np.save(f"labels_part_{part}.npy", np.array(y_true))
            print(f"💾 Progress saved (part {part}) - {len(y_true)} images processed.")
            part += 1
            y_true, y_scores = [], []

    except tf.errors.ResourceExhaustedError:
        BATCH_SIZE = max(8, BATCH_SIZE // 2)
        print(f"⚠️ OOM detected! Reducing batch size to {BATCH_SIZE} and continuing...")
        gc.collect()
        tf.keras.backend.clear_session()
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            VAL_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False
        ).map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=tf.data.AUTOTUNE)
        continue

if SAVE_PARTS:
    print("\nConsolidating saved parts...")
    preds_files = sorted([f for f in os.listdir() if f.startswith("preds_part_")])
    labels_files = sorted([f for f in os.listdir() if f.startswith("labels_part_")])
    for p, l in zip(preds_files, labels_files):
        y_scores.extend(np.load(p).tolist())
        y_true.extend(np.load(l).tolist())

y_true = np.array(y_true).astype(int)
y_scores = np.array(y_scores)

y_true = np.array(y_true, dtype=int).ravel()
y_scores = np.array(y_scores, dtype=float).ravel()

print(f"Total samples evaluated: {len(y_true)}")

print("\nCalculating best threshold...")
prec, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
f1_scores = np.nan_to_num(f1_scores)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

print(f"\nBest threshold found: {best_threshold:.4f}")
print(f"Precision: {prec[best_idx]:.4f}")
print(f"Recall: {recall[best_idx]:.4f}")
print(f"F1-score: {f1_scores[best_idx]:.4f}")

y_pred = (y_scores > best_threshold).astype(int)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

with open(SAVE_PATH, "w") as f:
    json.dump({"best_threshold": float(best_threshold)}, f)

plt.figure(figsize=(8, 6))
plt.plot(recall, prec, label="Precision-Recall Curve", color="blue")
plt.scatter(recall[best_idx], prec[best_idx], color="red", label=f"Best threshold = {best_threshold:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label="Fake (label=0)")
plt.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label="Real (label=1)")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best threshold = {best_threshold:.3f}")
plt.xlabel("Score (sigmoid)")
plt.ylabel("Frequency")
plt.title("Score Distribution by Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.close()

print(f"\n💾 Threshold saved at: {SAVE_PATH}")
print("📊 Charts saved: precision_recall_curve.png and score_distribution.png")
print("\n✅ Robust calibration completed successfully.")
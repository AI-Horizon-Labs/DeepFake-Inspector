import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve


print("TensorFlow versão:", tf.__version__)

train_dataset_path = '/workspace/datasets/Dataset/Train'
dataset_path = '/workspace/datasets/Dataset/Validation'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_path,
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary',  # binário: 0 para 'Fake', 1 para 'Real'
    shuffle=True,
)

print(train_ds.class_names)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary',
    shuffle=False,
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalização
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print("Real:", len(os.listdir(dataset_path + "/Real")))
print("Deepfakes:", len(os.listdir(dataset_path + "/Fake")))

labels = []
for _, label in train_ds.unbatch():
    labels.append(label.numpy().item())

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights_dict = dict(enumerate(class_weights))
print("📊 Class weights:", class_weights_dict)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True  # Fine-tuning total

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')  # saída binária (real ou deepfake)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])

# 📉 Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 📈 Callbacks para monitoramento e ajuste de aprendizado patience = 10
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-7)
]

# 🚀 Treinamento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# 🧪 Avaliação
y_scores = []
y_true = []
for images, labels in val_ds:
    preds = model.predict(images).flatten()
    y_scores.extend(preds)
    y_true.extend(labels.numpy().astype("int32"))

# 🎯 Otimização do threshold
prec, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
f1_scores = np.nan_to_num(f1_scores)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Melhor threshold: {best_threshold:.4f}")

# 📊 Classificação com threshold ótimo
y_pred = [1 if score > best_threshold else 0 for score in y_scores]
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
print(confusion_matrix(y_true, y_pred))

# 📈 Visualização dos scores
plt.hist(y_scores, bins=50)
plt.title("Distribuição dos scores (sigmoid)")
plt.xlabel("Score do modelo")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

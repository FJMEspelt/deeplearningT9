#!/usr/bin/env python
"""
06_data_augmentation.py

Entrenamiento con aumento de datos usando ImageDataGenerator sobre un modelo pre-entrenado especificado.
Este script:
  - Aplica aumentos (rotación, desplazamientos, zoom, flips) al set de entrenamiento.
  - Usa el modelo `base_cnn` importado de `src.model_zoo`.
  - Entrena con los generadores aumentados y de validación.
  - Guarda el modelo (.keras), el historial (metrics.json) y las gráficas de Loss/Accuracy.

Uso:
  python -m scripts.06_data_augmentation
"""
import datetime
import json
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

from src.config import IMG_SIZE, BATCH, TRAIN_DIR, TEST_DIR, MODELS, HISTORIES

# ---------------- Configuración ----------------
EPOCHS     = 10       # Número de épocas bajo para reducir carga computacional
MODEL_NAME = "base_cnn_aug"
PRETRAINED_MODEL_FILENAME = "augmented_cnn_tuned_20250607_172812.keras" # Nombre del modelo pre-entrenado a cargar
# ----------------------------------------------

# 1) Prepare data generators con augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

# 2) Cargar el modelo pre-entrenado configurado
model_path_to_load = MODELS / PRETRAINED_MODEL_FILENAME
print(f"Cargando modelo pre-entrenado: {model_path_to_load}")
model = load_model(model_path_to_load)

# 3) Entrenar
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# 4) Guardar artefactos
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Guardar modelo en formato .keras
MODELS.mkdir(parents=True, exist_ok=True)
model_path = MODELS / f"{MODEL_NAME}_{run_id}.keras"
model.save(model_path)

# Guardar historial metrics.json
HISTORIES.mkdir(parents=True, exist_ok=True)
hist_dir = HISTORIES / f"{MODEL_NAME}_{run_id}"
hist_dir.mkdir(parents=True, exist_ok=True)
with open(hist_dir / "metrics.json", "w") as f:
    json.dump(history.history, f, indent=2)

# 5) Plots de Loss y Accuracy
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout()
plt.savefig(hist_dir / "loss_curve.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout()
plt.savefig(hist_dir / "accuracy_curve.png", dpi=200)
plt.close()

print(f"Aumento de datos completado. Modelo: {model_path}")
print(f"Artefactos guardados en: {hist_dir}")

# 6) Evaluación final: matriz de confusión y classification report
# Reset del generador de validación para predecir desde el inicio
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen).argmax(axis=1)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Matriz de Confusión")
reports_fig = pathlib.Path("reports/figures")
reports_fig.mkdir(parents=True, exist_ok=True)
conf_path = reports_fig / f"{MODEL_NAME}_{run_id}_confusion_matrix.png"
plt.savefig(conf_path, dpi=200)
plt.close()
print(f"Confusion matrix saved to: {conf_path}")

# Classification report
report = classification_report(
    y_true, y_pred,
    target_names=list(val_gen.class_indices.keys())
)
reports_txt = pathlib.Path("reports")
reports_txt.mkdir(parents=True, exist_ok=True)
report_path = reports_txt / f"{MODEL_NAME}_{run_id}_classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to: {report_path}")
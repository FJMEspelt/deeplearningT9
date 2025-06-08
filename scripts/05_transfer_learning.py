#!/usr/bin/env python
"""
Transfer learning con ResNet50 y fine-tuning parcial ligero.

Esta rutina:
- Carga ResNet50 pre-entrenado en ImageNet (sin la parte superior).
- Congela las primeras capas del backbone.
- Añade un clasificador adaptado al nº de clases de tu dataset.
- Realiza fine-tuning de las últimas capas.
- Guarda el modelo (.keras), el historial (metrics.json) y las gráficas.

Uso (desde la raíz del proyecto):
    python -m scripts.05_transfer_learning
"""
import json
import datetime
import pathlib
import tensorflow as tf

from src.data_utils import make_generators
from src.data_utils import make_generators
import tensorflow as tf

from src.config import MODELS, HISTORIES

# ---------------- Configuración ----------------
HEAD_EPOCHS = 2        # Entrena solo la cabeza
FINE_TUNE_EPOCHS = 3   # Fine-tuning parcial
UNFREEZE_LAYERS = 10   # Últimas N capas a descongelar
MODEL_NAME = "resnet50_ft"
# -----------------------------------------------

def transfer_resnet50(input_shape=(150, 150, 3), num_classes=6):
    # Base model sin la capa superior
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    # Todas las capas por defecto se dejan entrenables; el freeze se gestiona en main()

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # Timestamp y directorios
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_dir = HISTORIES / f"{MODEL_NAME}_{run_id}"
    hist_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # Generadores
    train_gen, val_gen, _ = make_generators()
    num_classes = train_gen.num_classes

    # Construye el modelo
    model = transfer_resnet50(input_shape=(150,150,3), num_classes=num_classes)
    base = model.get_layer(name="resnet50")

    # --- Fase 1: cabeza solo ---
    for layer in base.layers:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=HEAD_EPOCHS
    )

    # --- Fase 2: fine-tuning parcial ---
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS
    )

    # Combina historiales
    metrics = {}
    for m in ["loss","val_loss","accuracy","val_accuracy"]:
        metrics[m] = history_head.history[m] + history_fine.history[m]

    # Guarda el modelo en formato .keras
    model_path = MODELS / f"{MODEL_NAME}_{run_id}.keras"
    model.save(model_path)

    # Guarda historial y gráficas
    with open(hist_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Gráfica de pérdida
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(metrics["loss"], label="train")
    plt.plot(metrics["val_loss"], label="val")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / "loss_curve.png", dpi=200)
    plt.close()

    # Gráfica de accuracy
    plt.figure()
    plt.plot(metrics["accuracy"], label="train")
    plt.plot(metrics["val_accuracy"], label="val")
    plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / "accuracy_curve.png", dpi=200)
    plt.close()

    print(f"Transfer learning completo. Modelo: {model_path}")
    print(f"Artefactos en: {hist_dir}")

if __name__ == "__main__":
    main()

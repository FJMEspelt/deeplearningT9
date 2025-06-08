#!/usr/bin/env python
"""
Entrena un modelo CNN ligero y guarda:
  • Pesos del modelo (.keras)
  • Historial de entrenamiento (metrics.json)
  • Gráficos de loss & accuracy (.png)
"""
import json, pathlib, datetime
import tensorflow as tf
from src.data_utils import make_generators
from src.train_utils  import run_training
import matplotlib.pyplot as plt
from src.config import MODELS, HISTORIES

# -------------------- Configuración --------------------
EPOCHS      = 10        # Nº de épocas de entrenamiento
MODEL_NAME  = "augmented_cnn"
EARLY_STOP_PATIENCE = 3   # Nº de épocas sin mejora antes de detener
# -------------------------------------------------------

def augmented_cnn(input_shape=(150, 150, 3), num_classes=None):
    """Versión más profunda del modelo base deep_cnn."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64]:  # menos bloques para aligerar
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)
    # Compilación — ajusta el loss a tu data (categorical vs sparse)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(history, out_dir):
    plt.figure()
    plt.plot(history.history["loss"],     label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"],     label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=200)
    plt.close()

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = HISTORIES / f"{MODEL_NAME}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    HISTORIES.mkdir(exist_ok=True)

    train_gen, val_gen, _ = make_generators()

    # Construye el modelo aumentado con el nº de clases correcto
    model_fn = lambda: augmented_cnn(num_classes=train_gen.num_classes)

    model, history = run_training(
        model_fn,
        train_gen,
        val_gen,
        model_name=MODEL_NAME,
        epochs=EPOCHS
    )

    # Guarda el modelo entrenado
    model.save(MODELS / f"{MODEL_NAME}_{run_id}.keras")

    # Persist history
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(history.history, f, indent=2)

    plot_history(history, out_dir)
    print(f"✔ Entrenamiento terminado. Artefactos en: {out_dir}")

if __name__ == "__main__":
    main()
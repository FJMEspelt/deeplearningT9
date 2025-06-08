#!/usr/bin/env python
"""
Hyper‑parameter tuning for an *ultra‑light* CNN (2 conv blocks, 3 trials × 3 epochs).

Search space
------------
* L1/L2 regularisation strength
* Kernel initialiser
* Activation function
* Use of Dropout per block
* Dropout rate
* Learning rate

Usage
-----
Just run:
    python -m scripts.04_hyperparam_tuning

All trial artefacts are saved under `tuner_logs/`.
"""

import datetime, json, pathlib
import tensorflow as tf
import keras_tuner as kt
from src.data_utils import make_generators
from src.config import MODELS, HISTORIES
import matplotlib.pyplot as plt

# ------------------------ Search space ------------------------ #
def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    reg_strength = hp.Float("reg_strength", 1e-5, 1e-2, sampling="log")
    regulariser = tf.keras.regularizers.l1_l2(l1=reg_strength, l2=reg_strength)

    kernel_init = hp.Choice(
        "kernel_init",
        ["he_normal", "he_uniform", "glorot_normal", "glorot_uniform"]
    )
    activation = hp.Choice("activation", ["relu", "gelu", "selu"])

    use_dropout = hp.Boolean("use_dropout")
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.4, step=0.1) if use_dropout else 0.0

    inputs = tf.keras.Input(shape=(150, 150, 3))
    x = inputs

    # Two lightweight convolutional blocks
    for filters in [32, 64]:
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same",
            kernel_initializer=kernel_init,
            kernel_regularizer=regulariser,
            activation=activation
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        if use_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        128,
        activation=activation,
        kernel_initializer=kernel_init,
        kernel_regularizer=regulariser
    )(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(6, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="augmented_cnn_tuned")

    lr = hp.Float("learning_rate", 1e-5, 3e-4, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
# -------------------------------------------------------------- #

def main():

    # Generators (same as previous scripts)
    train_gen, val_gen, _ = make_generators()

    tuner_dir = pathlib.Path("tuner_logs")
    tuner_dir.mkdir(exist_ok=True)

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name=datetime.datetime.now().strftime("aug_cnn_%Y%m%d_%H%M%S"),
        overwrite=False
    )

    tuner.search_space_summary()

    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        ]
    )

    tuner.results_summary()

    # Save best model and history
    best_model = tuner.get_best_models(num_models=1)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS / f"augmented_cnn_tuned_{timestamp}.keras"
    best_model.save(model_path)
    print(f"Best model saved to: {model_path}")

    # Retrieve training history from the best trial
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    hist_dict = {}
    for metric in ["loss", "val_loss", "accuracy", "val_accuracy"]:
    # history entries are MetricObservation objects; extract the `.value`
        hist_dict[metric] = [obs.value for obs in best_trial.metrics.get_history(metric)]

    hist_dir = HISTORIES / f"augmented_cnn_tuned_{timestamp}"
    hist_dir.mkdir(parents=True, exist_ok=True)

    with open(hist_dir / "metrics.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # Quick plots
    plt.figure()
    plt.plot(hist_dict["loss"], label="train")
    plt.plot(hist_dict["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / "loss_curve.png", dpi=200)

    plt.figure()
    plt.plot(hist_dict["accuracy"], label="train")
    plt.plot(hist_dict["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / "accuracy_curve.png", dpi=200)

    print("Ultra‑light tuning finished and artefacts saved.")

if __name__ == "__main__":
    main()
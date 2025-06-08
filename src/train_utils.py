import json, datetime
from pathlib import Path
from .config import MODELS, HISTORIES
from .config import FIGURES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def run_training(model_fn, train_gen, val_gen, model_name, epochs=20):
    model = model_fn()
    ckpt_path = MODELS / f"{model_name}.h5"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=ckpt_path, save_best_only=True)
    ]
    hist = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks)
    # save history
    hist_path = HISTORIES / f"{model_name}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(hist.history))
    
    fig_path = FIGURES / f"{model_name}_accuracy.png"
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy - {model_name}")
    plt.savefig(fig_path)
    plt.close()
    
    fig_path_loss = FIGURES / f"{model_name}_loss.png"
    plt.figure()
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss - {model_name}")
    plt.savefig(fig_path_loss)
    plt.close()

    # Predict on validation generator
    val_preds = model.predict(val_gen)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val_gen.classes

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = list(val_gen.class_indices.keys())
    fig_cm_path = FIGURES / f"{model_name}_confusion_matrix.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(fig_cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {fig_cm_path}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=False)
    report_path = FIGURES / f"{model_name}_classification_report.txt"
    report_path.write_text(report)
    print("Classification Report:")
    print(report)

    return model, hist

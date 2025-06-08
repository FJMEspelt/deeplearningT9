from src.data_utils import make_generators
from src.model_zoo import base_cnn
from src.train_utils import run_training
from src.config import IMG_SIZE, BATCH, SEED, CLASSES, TRAIN_DIR, TEST_DIR, MODELS, HISTORIES
import matplotlib.pyplot as plt
import json
from tensorflow.keras.models import load_model

force_train = False  # Cambiar a True para reentrenar

print("Generando generadores de datos...")
train_gen, val_gen, _ = make_generators()

model_path = MODELS / "base_model_trained.keras"
hist_path = HISTORIES / "base_model.json"
MODELS.mkdir(exist_ok=True)

print("Preparando modelo para entrenamiento o carga...")
if model_path.exists() and hist_path.exists() and not force_train:
    print("Cargando historial de entrenamiento y modelo guardado...")
    base_cnn = load_model(model_path)
    print(f"Modelo cargado desde: {model_path}")

    hist_dict = json.load(open(hist_path))
    plt.plot(hist_dict["accuracy"], label="train")
    plt.plot(hist_dict["val_accuracy"], label="val")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
else:
    print("Entrenando modelo desde cero...")
    model, hist = run_training(base_cnn, train_gen, val_gen,
                               model_name="base_model", epochs=15)
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
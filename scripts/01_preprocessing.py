import numpy as np
import cv2
import tqdm
from src.config import IMG_SIZE, DATA_PROC, TRAIN_DIR, TEST_DIR, CLASSES

print("Iniciando preprocesamiento de imágenes...")

DATA_PROC.mkdir(exist_ok=True)

def load_split(folder_path):
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        for img_path in tqdm.tqdm((folder_path / cls).glob("*.jpg"),
                                  desc=f"{folder_path.name}-{cls}"):
            im = cv2.imread(str(img_path))
            im = cv2.resize(im, IMG_SIZE)
            X.append(im)
            y.append(idx)
    print(f"{folder_path.name} procesado. Total imágenes: {len(X)}")
    return np.array(X), np.array(y)

splits = {
    "seg_train": TRAIN_DIR,
    "seg_test": TEST_DIR
}

for split_name, path in splits.items():
    x_path = DATA_PROC / f"{split_name}_X.npy"
    y_path = DATA_PROC / f"{split_name}_y.npy"
    if x_path.exists() and y_path.exists():
        print(f"Archivos {x_path.name} y {y_path.name} ya existen. Saltando procesamiento de {split_name}.")
        continue
    print(f"Procesando datos para: {split_name}...")
    X, y = load_split(path)
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Guardados: {x_path.name}, {y_path.name}")
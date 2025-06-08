#!/usr/bin/env python
"""
07_evaluation.py

Este script carga los historiales de validación de los modelos ya entrenados
y dibuja sus curvas de val_accuracy en un mismo gráfico para compararlas.

Uso:
    python -m scripts.07_evaluation
"""
import json
import pathlib
import os
import matplotlib.pyplot as plt
import re

from src.config import HISTORIES, FIGURES

plt.figure(figsize=(8, 5))

# Recolectar todas las fuentes de historial: JSON raíz y carpetas timestamped
history_sources = []
# JSON sueltos en HISTORIES/
for json_file in sorted(HISTORIES.glob("*.json")):
    label = re.sub(r'_[0-9]{8}_[0-9]{6}$', '', json_file.stem)
    history_sources.append((label, json_file))

# Carpetas timestamped con metrics.json
for d in sorted(HISTORIES.iterdir()):
    if d.is_dir():
        metrics_file = d / "metrics.json"
        if metrics_file.exists():
            label = re.sub(r'_[0-9]{8}_[0-9]{6}$', '', d.name)
            history_sources.append((label, metrics_file))

# Trazar cada historia
for label, hist_file in history_sources:
    print(f"Procesando '{label}' desde {hist_file.name}")
    metrics = json.load(open(hist_file, "r"))
    val_acc = metrics.get("val_accuracy")
    if not val_acc:
        print(f"⚠ 'val_accuracy' no está en {hist_file}.")
        continue
    epochs = range(len(val_acc))
    plt.plot(epochs, val_acc, marker='o', label=label)

plt.title("Comparativa val_accuracy por modelo")
plt.xlabel("Época")
plt.ylabel("Validación Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Crea carpeta de figuras si no existe
FIGURES.mkdir(parents=True, exist_ok=True)
out_path = FIGURES / "comparison_val_accuracy.png"
plt.savefig(out_path, dpi=200)
print(f"Gráfico guardado en: {out_path}")
plt.show()
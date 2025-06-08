#!/usr/bin/env python
"""
08_evaluation_table.py

Evalúa todos los modelos entrenados en el conjunto de test y muestra una tabla comparativa 
de Test Accuracy con coloreado de celdas según el valor.

Uso:
    python -m scripts.08_evaluation_table
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import pathlib
import re

from src.data_utils import make_generators
from src.config import MODELS, FIGURES

# Asegurar dirs de salida
FIGURES.mkdir(parents=True, exist_ok=True)

# Generador de test
_, _, test_gen = make_generators()

# Evaluar cada modelo guardado en MODELS/
results = []
model_files = sorted(MODELS.glob("*.keras"))
for model_path in model_files:
    raw_label = model_path.stem
    # Eliminar sufijo _YYYYMMDD_HHMMSS del nombre
    label = re.sub(r'_[0-9]{8}_[0-9]{6}$', '', raw_label)
    print(f"▶ Evaluando {label} desde {model_path.name}")
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(test_gen, verbose=0)
    results.append({"Model_Name": label, "Test_Accuracy": round(acc, 3)})

# Construir DataFrame y ordenar
df = pd.DataFrame(results)
df = df.sort_values("Test_Accuracy", ascending=False).reset_index(drop=True)

# Mostrar en consola
print("\nComparativa de Test Accuracy:")
print(df.to_string(index=False))

# Crear figura de tabla coloreada
fig, ax = plt.subplots(figsize=(12, len(df)*0.6 + 1))
ax.axis("off")
tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center"
)

# Colorear la columna Test_Accuracy
norm = colors.Normalize(vmin=df["Test_Accuracy"].min(), vmax=df["Test_Accuracy"].max())
cmap = plt.get_cmap("coolwarm")
for i, acc in enumerate(df["Test_Accuracy"], start=1):
    tbl[(i, 1)].set_facecolor(cmap(norm(acc)))

tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.5, 1.5)

plt.title("Test Accuracy por Modelo", pad=20)
out_path = FIGURES / "comparison_test_accuracy_table.png"
plt.savefig(out_path, bbox_inches="tight", dpi=200)
print(f"✔ Tabla guardada en: {out_path}")

# Además, guardar la tabla en reports/figures
reports_fig = pathlib.Path("reports/figures")
reports_fig.mkdir(parents=True, exist_ok=True)
report_out = reports_fig / "comparison_test_accuracy_table.png"
plt.savefig(report_out, bbox_inches="tight", dpi=200)
print(f"Tabla guardada también en: {report_out}")
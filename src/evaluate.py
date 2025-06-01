import json, pandas as pd, tensorflow as tf
from pathlib import Path
from .config import MODELS, HISTORIES
from .data_utils import make_generators

def main():
    _, _, test_gen = make_generators()
    results = []
    for model_file in MODELS.glob("*.h5"):
        model = tf.keras.models.load_model(model_file)
        loss, acc = model.evaluate(test_gen, verbose=0)
        results.append({"Model_Name": model_file.stem, "Test_Accuracy": round(acc,4)})
    df = pd.DataFrame(results).sort_values("Test_Accuracy", ascending=False)
    print(df.to_markdown(index=False))
    df.to_csv("reports/test_results.csv", index=False)

if __name__ == "__main__":
    main()

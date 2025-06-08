#!/usr/bin/env python
"""
Quick dataset inspection:
  • Counts images per class in train/test splits
  • Shows a bar chart of training-set class counts
  • Displays one random training image for a manual sanity check
"""

from pathlib import Path
from collections import Counter
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

DATA_RAW = Path("/Users/javiermolinaespelt/Documents/Master IA/deeplearningT9/data/raw")
TRAIN_DIR = DATA_RAW / "seg_train" / "seg_train"
TEST_DIR = DATA_RAW / "seg_test" / "seg_test"

# ---------------------------------------------------------------------
# Helper: count JPGs in each class directory
# ---------------------------------------------------------------------
def count_images(split_dir: Path) -> dict[str, int]:
    return {
        cls: len(list((split_dir / cls).glob("*.jpg")))
        for cls in os.listdir(split_dir)
        if (split_dir / cls).is_dir()
    }

def main() -> None:
    print(f"Using training data from: {TRAIN_DIR}\nUsing test data from: {TEST_DIR}")
    # 1) Basic stats ---------------------------------------------------
    train_counts = count_images(TRAIN_DIR)
    test_counts  = count_images(TEST_DIR)

    print("Train:", train_counts)
    print("Test :", test_counts)

    # 2) Bar plot of training counts ----------------------------------
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=list(train_counts.keys()),
        y=list(train_counts.values())
    )
    plt.title("Images per class (train)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 3) Visual sanity check ------------------------------------------
    cls = random.choice(list(train_counts.keys()))
    sample_path = random.choice(list((TRAIN_DIR / cls).glob("*.jpg")))
    img = Image.open(sample_path)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Random sample: {cls}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
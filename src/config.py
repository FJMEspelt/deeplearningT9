from pathlib import Path

# --- paths ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path("/Users/javiermolinaespelt/Documents/Master IA/deeplearningT9/data/raw/seg_train/seg_train")
TEST_DIR  = Path("/Users/javiermolinaespelt/Documents/Master IA/deeplearningT9/data/raw/seg_test/seg_test")
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "saved_models"
HISTORIES = ROOT / "histories"
FIGURES = ROOT / "reports" / "figures"

IMG_SIZE = (150, 150)
BATCH   = 32
SEED    = 42
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
N_CLASSES = len(CLASSES)

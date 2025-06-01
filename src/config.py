from pathlib import Path

# --- paths ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "saved_models"
HISTORIES = ROOT / "histories"
FIGURES = ROOT / "reports" / "figures"

IMG_SIZE = (150, 150)
BATCH   = 32
SEED    = 42
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
N_CLASSES = len(CLASSES)

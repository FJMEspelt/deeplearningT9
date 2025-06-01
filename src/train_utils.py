import json, datetime
from pathlib import Path
from .config import MODELS, HISTORIES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def run_training(model_fn, train_gen, val_gen, model_name, epochs=20):
    model = model_fn()
    ckpt_path = MODELS / f"{model_name}.h5"
    callbacks  = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=ckpt_path, save_best_only=True)
    ]
    hist = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks)
    # save history
    hist_path = HISTORIES / f"{model_name}.json"
    hist_path.write_text(json.dumps(hist.history))
    return model, hist

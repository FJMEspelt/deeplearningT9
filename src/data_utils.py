from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH = 32
SEED = 42
TRAIN_DIR = "/Users/javiermolinaespelt/Documents/Master IA/deeplearningT9/data/raw/seg_train/seg_train"
TEST_DIR = "/Users/javiermolinaespelt/Documents/Master IA/deeplearningT9/data/raw/seg_test/seg_test"
from pathlib import Path

def make_generators(validation_split=0.20, augment=False):
    """Return train_gen, val_gen, test_gen (Keras generators)."""
    if augment:
        train_kwargs = dict(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split)
    else:
        train_kwargs = dict(rescale=1./255, validation_split=validation_split)

    train_datagen = ImageDataGenerator(**train_kwargs)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="training",
        seed=SEED)

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation",
        seed=SEED)

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False)

    return train_gen, val_gen, test_gen

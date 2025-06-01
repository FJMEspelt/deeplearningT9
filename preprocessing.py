from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (150, 150)
BATCH = 32
train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.20
)
train_ds = train_datagen.flow_from_directory(
    "Seg_train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training", seed=42)
val_ds = train_datagen.flow_from_directory(
    "Seg_train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation", seed=42)
test_datagen = ImageDataGenerator(rescale=1./255)
test_ds = test_datagen.flow_from_directory(
    "Seg_test",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False)

import tensorflow as tf
from tensorflow.keras import layers, models
from .config import IMG_SIZE, N_CLASSES

def base_cnn():
    m = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE,3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(N_CLASSES, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def deep_cnn():
    m = models.Sequential()
    for filters in (32, 64, 128):
        m.add(layers.Conv2D(filters, (3,3), activation='relu', padding='same',
                            input_shape=(*IMG_SIZE,3) if filters==32 else None))
        m.add(layers.Conv2D(filters, (3,3), activation='relu', padding='same'))
        m.add(layers.MaxPooling2D())
        m.add(layers.Dropout(0.25))
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(N_CLASSES, activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# -- hyperparameter tuning model is built inside the tuner (see 04 notebook)
# -- transfer-learning builder (MobileNetV2 example)

def fine_tune_mobilenet(freeze_until=100):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE,3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(base.input, out)

    # first pass: freeze backbone
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

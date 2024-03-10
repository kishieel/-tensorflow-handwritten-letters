import pathlib

import tensorflow as tf
import numpy as np

data_dir = pathlib.Path('/data').with_suffix('')
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(100, 100),
    batch_size=32
)
valid_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(100, 100),
    batch_size=32
)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = valid_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

normalized_ds = train_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(100, 100, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
)

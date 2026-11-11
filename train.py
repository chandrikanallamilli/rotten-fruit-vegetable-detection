import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG, BATCH = (224, 224), 32
train = tf.keras.utils.image_dataset_from_directory("dataset/train", image_size=IMG, batch_size=BATCH)
val = tf.keras.utils.image_dataset_from_directory("dataset/val", image_size=IMG, batch_size=BATCH)
print("Classes:", train.class_names)          # ['fresh', 'rotten']

base = MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False                          # transfer learning: freeze pretrained layers

model = models.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),
    layers.Rescaling(1.0 / 127.5, offset=-1),   # MobileNetV2 preprocessing
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),      # 0 = fresh, 1 = rotten
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)
model.save("model/fruit_model.keras")
print("Saved model/fruit_model.keras")

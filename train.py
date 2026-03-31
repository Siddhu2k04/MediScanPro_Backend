# ===== FIX FOR CORRUPTED IMAGES =====
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== IMPORTS =====
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===== SETTINGS =====
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# ===== DATA GENERATORS =====
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===== TRANSFER LEARNING MODEL =====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False
 
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(train_data.num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ===== COMPILE =====
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===== TRAIN =====
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ===== SAVE MODEL =====
model.save("fracture_model.h5")

print("✅ MODEL TRAINING COMPLETED")



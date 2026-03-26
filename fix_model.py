import tensorflow as tf

# load your original model
model = tf.keras.models.load_model("ct_scan_model.h5")

# save it again in compatible format
model.save("fixed_model.h5")

print("Model fixed and saved!")
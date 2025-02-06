import tensorflow as tf

# Load your HDF5 model
model = tf.keras.models.load_model("efficientnetb0_skin_conditions.h5")

# Save in new Keras format
model.save("efficientnetb0_skin_conditions.keras", save_format="keras")

print("Conversion complete! 🎉")

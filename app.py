import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("D:/ALL/DermaNet/efficientnetb0_skin_conditions.keras")  # Update with your actual model path

# Define class labels (Updated to match your dataset)
class_labels = [
    "Acne",
    "Alopecia",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Eczema",
    "Hemangioma",
    "Hidradenitis Suppurativa",
    "Hives",
    "Keratosis",
    "Lupus",
    "Moles and Skin Cancer",
    "Nail Disease",
    "Pemphigus",
    "Psoriasis",
    "Rosacea",
    "Scabies",
    "Skin Allergy",
    "Sun Damaged",
    "Tick Bite",
    "Vasculitis",
    "Venous",
    "Vitiligo",
    "Warts"
]

st.title("DermaNet - Skin Disease Classifier 🩺✨")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display result
    st.write(f"**Prediction:** {class_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2%}")

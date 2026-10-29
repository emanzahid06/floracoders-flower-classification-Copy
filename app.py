
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="FloraCoders",
    page_icon="🌸",
    layout="centered"
)

# Website heading
st.title("🌸 FloraCoders")
st.subheader("AI-Based Flower Species Classification")
st.write("Upload a flower image and let our AI model identify it!")

# Load trained model
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model(
"floracoders_best_model.keras"
    )
model = load_flower_model()

# Flower classes in the correct model order
class_names = [
    "dandelion",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]

# Image uploader
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Your Uploaded Flower", use_container_width=True)

    if st.button("Identify Flower 🌼"):
        image = image.resize((180, 180))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = np.argmax(predictions)

        st.success(
            f"Predicted Flower: {class_names[predicted_index].title()}"
        )

        confidence = float(predictions[predicted_index]) * 100
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Prediction Scores")

        for i, flower in enumerate(class_names):
            st.write(f"{flower.title()}: {predictions[i] * 100:.2f}%")

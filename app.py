
import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Load CNN Model
# -----------------------------

model = load_model("road_damage_model.h5", compile=False)

# -----------------------------
# Class Labels
# -----------------------------

class_names = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
}

# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

# -----------------------------
# App Title
# -----------------------------

st.title("CNN-Based Road Damage Detection System")

st.write(
    "Upload a road image to detect road damage category using CNN."
)

# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction Function
# -----------------------------

def predict_image(image):

    img = np.array(image)

    resized = cv2.resize(img, (128,128))

    normalized = resized / 255.0

    reshaped = np.reshape(
        normalized,
        (1,128,128,3)
    )

    prediction = model.predict(reshaped)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return predicted_class, confidence

# -----------------------------
# Run Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    if st.button("Predict Road Damage"):

        predicted_class, confidence = predict_image(image)

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {class_names[predicted_class]}"
        )

        st.subheader("Confidence Score")

        st.info(
            f"{confidence:.2f}"
        )

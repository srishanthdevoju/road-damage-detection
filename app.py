
import streamlit as st
import numpy as np
import cv2

from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# ---------------------------------
# Rebuild CNN Architecture
# ---------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

# ---------------------------------
# Load ONLY weights
# ---------------------------------

model.load_weights("road_damage.weights.h5")

# ---------------------------------
# Class Labels
# ---------------------------------

class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

# ---------------------------------
# Streamlit UI
# ---------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

st.title("🚧 Road Damage Detection")

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

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

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):

        predicted_class, confidence = predict_image(image)

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}"
        )

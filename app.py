
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
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

# ---------------------------------
# REBUILD CNN MODEL
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
# LOAD MODEL WEIGHTS
# ---------------------------------

model.load_weights("road_damage.weights.h5")

# ---------------------------------
# CLASS LABELS
# ---------------------------------

class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

# ---------------------------------
# HEADER
# ---------------------------------

st.title("🚧 AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

# ---------------------------------
# ABOUT PROJECT
# ---------------------------------

st.header("📘 About the Project")

st.write("""
Road monitoring is important for transportation safety,
vehicle protection, and smart city infrastructure management.

This system uses Convolutional Neural Networks (CNNs)
to automatically classify road damages from uploaded images.

Applications include:
- Smart city monitoring
- Highway inspection
- Municipal maintenance systems
- AI-based infrastructure analysis
""")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------

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

# ---------------------------------
# RUN PREDICTION
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.header("🖼 Uploaded Image Preview")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🚀 Predict"):

        predicted_class, confidence = predict_image(image)

        damage_type = class_names[predicted_class]

        st.header("📊 Prediction Result")

        st.success(
            f"Prediction: {damage_type}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

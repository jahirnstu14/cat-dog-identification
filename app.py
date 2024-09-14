import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cats_dogs_cnn_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (256, 256))  # Resize to match model input
    img_resized = img_resized.reshape(1, 256, 256, 3)  # Add batch dimension
    img_resized = img_resized.astype('float32') / 255.0  # Normalize
    return img_resized

# Function to predict if the image is a dog or not
def predict(image):
    preprocessed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data[0][0] > 0.5:  # Threshold can be adjusted
        return "Dog"
    else:
        return "Cat"

# Streamlit interface
st.title("Dog or Cat Identification")

# Set custom background color using Streamlit's markdown and CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #FF7F50, #1E90FF, #32CD32);
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        st.write("Identification:")
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")

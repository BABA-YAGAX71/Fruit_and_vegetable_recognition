import streamlit  as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os 
from PIL import Image


#Tensorflow model
def model_prediction(test_image):
    # Use PIL to open the file stream directly
    image = Image.open(test_image) 
    image = image.resize((64, 64)) # Resize the PIL image
    model = tf.keras.models.load_model("trained_model.h5")
    image = keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return np.argmax(predictions, axis=1)[0]

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About project", "Prediction"])


#Main Page

if app_mode == "Home":
    st.header("Fruits and vegetables recognition system")
    st.image("home_image.jpg")

#about project page
if app_mode == "About project":
    st.header("About project")
    st.subheader("""About dataset""")
    st.text("Welcome to the Fruit Ripeness Classification Dataset! This comprehensive dataset is designed for researchers, data scientists, and machine learning enthusiasts interested in fruit classification based on ripeness stages.")
    st.subheader("content")
    st.text("The dataset is organized into three main folders:")
    st.text("Train: Contains 100 images per category.")
    st.text("Test: Contains 10 images per category.")
    st.text("Validation: Contains 10 images per category.")

#Prediction page
if app_mode == "Prediction":
    st.header("Model Prediction")
    st.text("Upload an image of a fruit or vegetable, and the model will predict the fruit or vegetable")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        st.image(file)
        st.text("File uploaded successfully")
        if st.button("Predict"):
            st.text("Model has predicted the image as :")
            result_index = model_prediction(file)
            #Reading  Labels
            with open("labels.txt") as f:
                label = [line.strip() for line in f.readlines() if line.strip()]



            st.write(f"Loaded {len(label)} labels.")
            st.write(f"Predicted Index: {result_index}")
            st.success("Model is predicting its a {}".format(label[result_index]))   
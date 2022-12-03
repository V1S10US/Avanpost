import time

import requests
import streamlit as st
from PIL import Image

PORT = 8080

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Avanpost transfer learning challenge")
st.image(Image.open('D:\Programming\Projects\Avanpost\eND6lE57NIs.jpg'))

'''Adding new class to pre trained model'''
if st.button("Add class"):
    new_class = st.text_input('Enter new class')
    if new_class is not None:
        st.success('The current class is', new_class)
        files = {"class_name": new_class}
        res = requests.post(f"http://backend:{PORT}/add_class", files=files)
        status = res.json().get("status")
        st.write(status)


'''Predict class by image'''
if st.button("Predict"):

    new_image = st.file_uploader("Choose an image", key=2)
    print(new_image)
    st.image(new_image)
    if new_image is not None:
        files = {"file": new_image.getvalue()}
        res = requests.post(f"http://backend:{PORT}/predict", files=files)
        img_path = res.json()
        predicted_name = img_path.get("name")
        st.success('Predicted class', predicted_name)
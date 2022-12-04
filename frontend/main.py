import streamlit as st
import requests
import json
import pandas as pd


import requests
import streamlit as st
from PIL import Image


st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Test web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image", type=['jpg', 'png'])

# displays a button
if st.button("Predict"):
    if image is not None:
        files = {"file": image.getvalue()}
        res = requests.post("http://127.0.0.1:8080/predict", files=files)
        predict = res.text
        st.success(res.text)


if st.button("Predict mulitple"):
    if image is not None:
        zp = st.file_uploader("Choose dataset", type=['zip'])
        files = {"file": image.getvalue()}
        res = requests.post("http://127.0.0.1:8080/predict_mulitple", files=files)
        csv_file = res.text
        st.success(res.text)
# df = pd.read_csv("/home/nap-time/HousePriceKazan/House_data.csv")

# def front_end():
#     st.title("Цена дома в городе Казань")

#     rooms = st.number_input("Количество комнат")
#     square = st.number_input("Плошадь")
#     floor = st.number_input("Этаж")
#     total_floor = st.number_input("Максимальный этаж")
#     metro_station = st.selectbox("Ближайшая станция метро", df['metro_station'].unique())
#     time_to_metro = st.number_input("Путь до метро(в минутах)")
#     transport = st.selectbox("Пешком или на транспорте", df.transport.unique())

#     item = {
#         'rooms' : rooms,
#         'square': square,
#         'floor': floor,
#         'total_floor':total_floor,
#         'metro_station': metro_station,
#         'time_to_metro':time_to_metro,
#         'transport': transport,
#         'square_room' : square / rooms,
#         'floor_coef' : floor / total_floor
#     }

#     if st.button("Predict"):
#         res = requests.post("http://0.0.0.0:8080/predict", json=item)
#         predict = res.text
#         st.success(f"Цена: {predict}")
# if __name__ == '__main__':
#     front_end()


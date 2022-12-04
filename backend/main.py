import cv2
import uvicorn
from pydantic import BaseModel
from glob import glob
from fastapi import FastAPI
from fastapi import UploadFile, File
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
import numpy as np
import io
from PIL import Image
import zipfile
import os
from tensorflow import keras
import csv

app = FastAPI()

MODEL_PATH = "D:/Programming/Projects/Avanpost/model"
tags = ['bicycle', 'dumper', 'horse', 'lawn_mower', 'ski', 'snowboard', 'tractor', 'train', 'truck', 'van']
data_path = 'D:/Programming/Projects/Avanpost/Image/'
#tags ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ
class_num = len(tags)
IMG_SIZE = (224, 224) # размер входного изображения сети



'''archive = 'file.zip'
with zipfile.ZipFile(archive, 'r') as zip_file:
    zip_file.extractall(directory_to_extract_to)'''

def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, target_size)
    return resnet50.preprocess_input(img) 

def predict_preprocess(label_name, path, target_size=IMG_SIZE):

    num_files = len(os.listdir(path))

    res_array = []

    for img in os.listdir(path):
        path_img = path + img
        res_array.append(load_image(path_img))

    res_array = np.array(res_array)
    res_labels = np.array([label_name * res_array.shape[0]])

    return res_array, res_labels

def preprocess(tags, classes_count):
    x = []
    y = []
    for i, tag in enumerate(tags):
        files = glob(data_path + tag + '/*.jpg')
        for path in files:
            try:
                x.append(load_image(path))
                y.append(i)
            except:
                print(path)

    y = np.array(y)
    x = np.array(x)
    y = keras.utils.to_categorical(y, classes_count)
    return x, y


X, y = preprocess(tags, class_num)




# Достаем нашу модель
model = keras.models.load_model(MODEL_PATH)


# Информация о том, что всё успешно запустилось
@app.get('/')
def root():
    return {"message": "Active"}

class Item(BaseModel):
    name: str


def ans(pred):
  summ = 0
  answ = []
  while summ < 0.3:
    answ, buf = search(pred, answ)
    summ += buf
  return answ

def search(pred, inds):
  maxel = 0
  ind = 0
  for i in range(len(pred)):
    if pred[i] > maxel and (i not in inds):
      maxel = pred[i]
      ind = i
  inds.append(ind)
  return inds, maxel

# Просто предикт
@app.post("/predict")
def predict(file: bytes=File(...)):

    image = np.array(Image.open(io.BytesIO(file)).convert('RGB'))
    image_resized = cv2.resize(image, (224, 224))
    image = np.expand_dims(image_resized, axis=0)
    pred = model.predict(image)
    ans = []
    summ = 0
    while summ < 0.9:
        ans.append(np.argmax(pred))
        summ += pred[0, np.argmax(pred)]
        pred[0, np.argmax(pred)] = 0

    output = []
    for i in ans:
        output.append(tags[i])

    return ';'.join(output)


@app.post("/predict_multiple")
def predict_multiple(file: bytes=File(...)):

    directory = './'
    with zipfile.ZipFile(file, 'r') as zip_file:
        zip_file.extractall(directory)

    with open('predict.csv', 'w+', newline='') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = [f'predict{i}' for i in range(1, 11)]
        row.insert(0, 'Id')
        spamwriter.writerow(row)
        id = 0
        for f_path in os.listdir(file):

            f = directory + f_path
            image = np.array(Image.open(io.BytesIO(f)).convert('RGB'))
            image_resized = cv2.resize(image, (224, 224))
            image = np.expand_dims(image_resized, axis=0)
            pred = model.predict(image)
            ans = []
            summ = 0
            while summ < 0.4:
                ans.append(np.argmax(pred))
                summ += pred[0, np.argmax(pred)]
                pred[0, np.argmax(pred)] = 0

            output = []
            for i in ans:
                output.append(tags[i])


            spamwriter.writerow([id, *output])
            id += 1

        return 

# Добавить класс
@app.post("/add_class")
def add_class(class_name: str):
    global tags, class_num, X, y, model
    tags.append(class_name)
    class_num = len(tags)

    temp_X, temp_y = preprocess([class_name], class_num)
    temp_y = temp_y[:, ::-1]
    # Concatenate X with new images
    X = np.concatenate((X, temp_X), axis=0)

    # Add 0 to every row(one-hot)
    padd = np.zeros((1, y.shape[0]))
    padd_y = np.concatenate((y, padd.T), axis=1)

    # Concatenate y
    y = np.concatenate((padd_y, temp_y), axis=0)

    
    new_model = keras.Sequential()
    for layer in model.layers[:-3]:
        new_model.add(layer)

    for layer in new_model.layers:
        layer.trainable=False

    new_model.add(model.layers[-3])
    new_model.add(model.layers[-2])
    new_model.add(Dense(class_num, activation='softmax'))

    new_model.compile(loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-3), metrics=['accuracy'])

    new_model.fit(X, y, batch_size=10, epochs=10, verbose=1, shuffle = True)

    model = new_model

    return 'Success'





if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, host='127.0.0.1', reload=True)
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

data_path = 'D:/Programming/Projects/Avanpost/Dataset200/'
tags = ['bicycle', 'dumper', 'horse', 'lawn_mower', 'ski', 'snowboard', 'tractor', 'train', 'truck', 'van']
class_num = len(tags)
IMG_SIZE = (224, 224) # размер входного изображения сети



MODEL_PATH = "D:/Programming/Projects/Avanpost/model"
model = keras.models.load_model(MODEL_PATH)

def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, target_size)
    return resnet50.preprocess_input(img)


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




#global class_num, X, y, model

class_name = 'skateboard'

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
    layer.trainable = False

new_model.add(model.layers[-3])
new_model.add(model.layers[-2])
new_model.add(Dense(class_num, activation='softmax'))

new_model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=1e-3), metrics=['accuracy'])

new_model.fit(X, y, batch_size=10, epochs=3, verbose=1, shuffle=True)

model = new_model

model.save('D:/Programming/Projects/Avanpost/new_model')
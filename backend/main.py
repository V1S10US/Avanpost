import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image


from frontend.main import PORT
import config
import inference

import os
import gi_scraper

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

def new_fit(): pass

@app.post("/add_class")
def add_new_class(file: UploadFile = File(...)):

    class_name = file.class_name  # get name from message
    samples_amount = 200
    # creating Scraper object
    scraper = gi_scraper.Scraper(process_count=1)
    # use scrape method to fire queries - returns ScrapedResponse object
    scraped_response = scraper.scrape(class_name, timeout=30, count=samples_amount)

    new_class_dir = f"./Dataset/{class_name}"
    os.mkdir(new_class_dir)
    scraped_response.write(path="./Dataset", filename="query").download(path=new_class_dir, thread_count=1)
    # get returns a dictionary with metadata and list of scraped urls
    # can be chained only at the end of the chained methods (write and download)
    scraped_response.get()

    # call close method or (del scraper) once scraping is done
    # needed for avoiding program going into an infinite loop
    scraper.close()

    new_fit()  # pass new samples to pre-trained model
    return {"status": 'ok'}

@app.post("/predict")
def predict_by_image(file: UploadFile = File(...)):

    image = np.array(Image.open(file.file)) # get image from message

    class_label = 'horse'  # change to new_fit(image) # predict label for image

    # name = file.file.filename
    return {"class_name": class_label}  # return label like message


async def combine_images(output, resized, name):
    final_image = np.hstack((output, resized))
    cv2.imwrite(name, final_image)


@app.post("/{style}")
async def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = config.STYLES[style]
    start = time.time()
    output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    print(f"name: {name}")
    # name = file.file.filename
    cv2.imwrite(name, output)
    models = config.STYLES.copy()
    del models[style]
    asyncio.create_task(generate_remaining_models(models, image, name))
    return {"name": name, "time": time.time() - start}


async def generate_remaining_models(models, image, name: str):
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    await event_loop.run_in_executor(
        executor, partial(process_image, models, image, name)
    )


def process_image(models, image, name: str):
    for model in models:
        output, resized = inference.inference(models[model], image)
        name = name.split(".")[0]
        name = f"{name.split('_')[0]}_{models[model]}.jpg"
        cv2.imwrite(name, output)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

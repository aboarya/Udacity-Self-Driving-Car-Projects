import os
import json
import pickle

import cv2

from keras.models import model_from_json

global model

_location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

model_file = os.path.join(_location__, 'outputs/robot.json')
weights_file = os.path.join(_location__, "weights-improvement-199-0.0186.h5")

with open(model_file, 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")

model.load_weights(weights_file)

def predict(frame):
    img_height, img_width = 64, 64

    image = cv2.imdecode(frame, 3)

    image_array = cv2.resize(image, (64, 64))

    image_array = image_array / 255.

    img = image_array.reshape((1, img_height, img_width, 3))

    return model.predict(img, batch_size=1)

import json
import pickle

import cv2
import numpy as np

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 112, 128

# load track images unseen to the model
data = pickle.load(open('/Users/sam.mohamed/robot-test-track.p', 'rb'))

mean = np.mean(np.array([[item['left'], item['right']] for k, item in data.items()])) 

X = np.array([cv2.cvtColor(item['rgb_image'], cv2.COLOR_BGR2GRAY).astype('float64').reshape((1, img_height, img_width, 1)) for k, item in data.items()])

X[0] = X[0]

model_file = '/Users/sam.mohamed/outputs/robot_navigation/robot_navigation.json'

with open(model_file, 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")
weights_file = model_file.replace('json', 'hd5')
model.load_weights(weights_file)

for image in list(X):
	print(float(model.predict(image, batch_size=1)))
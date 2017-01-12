
# coding: utf-8

# In[1]:

import os
import pickle
import json
import random
from PIL import Image
from io import BytesIO


import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

from sklearn.model_selection import train_test_split


# In[2]:


img_height, img_width = 180, 240

nb_epoch = 20
batch_size = 100

data = pickle.load(open('robot-S-track.p', 'rb'))

X = list()
Y = list()

#mean = np.mean(np.array([round(item['left'], 5) - round(item['right'],5) for k, item in data.items()])) 



# In[3]:

for k,item in data.items():
    image = Image.open(BytesIO(item['img']))
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X.append(img)
    Y.append([round(item['left'], 5), round(item['right'], 5)])


# In[4]:

X = np.array(X)
Y = np.array(Y)


# In[5]:

X = X.reshape((-1, img_height, img_width, 1))


# In[6]:

Y = [[round(item['left'], 5) , round(item['right'], 5)] for k, item in data.items()]

#Y = np.array([num / mean for num in Y])



# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

traingen = ImageDataGenerator(rescale=1./255.)

traingen.fit(X_train)

train_generator = traingen.flow(X_train, y_train, batch_size=batch_size)

testgen = ImageDataGenerator(rescale=1./255)

testgen.fit(X_test)
test_generator = testgen.flow(X_test, y_test, batch_size=batch_size)


# In[ ]:


model = Sequential()
# ((240 - 8) * 4) / 4)
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=(img_height, img_width, 1)))
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 4, 4, subsample=(2, 2)))
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))


adam = Adam(lr=0.0001)

model.compile(loss='mean_squared_error', optimizer=adam)

model.fit_generator(
        train_generator,
        samples_per_epoch=len(X_train),
        nb_epoch=nb_epoch,
        validation_data=test_generator,
        nb_val_samples=len(X_test))
  



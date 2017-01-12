import os
import pickle
import json
import random
import csv

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import Callback
from keras.utils import np_utils

_index_in_epoch = 0

nb_epoch = 500

batch_size = 32

img_height, img_width = 64, 64

global threshold
threshold = 0.04

class LifecycleCallback(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = threshold + (i * .05) 
        if threshold > 0.8:
            threshold = 0.2


def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)


def load_training_and_validation():

    data = pickle.load(open('robot-balanced-data.p', 'rb'))

    X_train = data['X_train']
    y_train = data['Y_train']

    X_test = data['X_val']
    y_test = data['Y_val']

    X_train, Y_train = shuffle(X_train, y_train)
    
    X_test, Y_test = shuffle(X_test, y_test)

    return X_train, Y_train, X_test, Y_test

def resize_image(img):
   return cv2.resize(img,( 64, 64))  

def affine_transform(img, pixels, controls, adjust, right=True):

    cols, rows, ch = img.shape
        
    pts1 = np.float32([[10,10], [200,50], [50,250]])

    if right:
        pts2 = np.float32([[10, 10], [200+pixels, 50], [50, 250]])
        controls[1] = controls[1] + adjust

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (rows, cols))

    return dst.reshape((cols, rows, ch)), controls


def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        data, labels = shuffle(data, labels)
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]


def transform_generator(x, y, batch_size=32, is_validation=False):
    while True:
        images, labels = list(), list()

        _images, _labels = next_batch(x, y, batch_size)

        current = os.path.dirname(os.path.realpath(__file__))

        # read in images as grayscale
        # affine transform (right and left)
        # to add additional angles
        for i in range(len(_images)):
            img =_images[i]

            resized = resize_image(img)

            # img = img.reshape(img_height, img_width, 3)
            images.append(resized)
            controls = list(_labels[i])
            labels.append(controls)

            if is_validation: continue

            angle = abs(controls[1] - controls[0])
            # print("angle", angle, controls)

            if threshold > angle:
                pixels = 15
                adjust = .001 * pixels * threshold
            
                image, controls = affine_transform(img, pixels, controls, adjust, right=True)
                
                resized = resize_image(image)
                images.append(resized)
                labels.append(controls)
                # print(adjust, controls)

                pixels = 20
                adjust = .001 * pixels * threshold
            
                image, controls = affine_transform(img, pixels, controls, adjust, right=True)
                resized = resize_image(image)
                images.append(resized)
                labels.append(controls)
                # print(adjust, controls)


        X = np.array(images, dtype=np.float64).reshape((-1, img_height, img_width, 3))

        X /= 255.

        Y = np.array(labels, dtype=np.float64)

        # raise RuntimeError(bad)

        yield (X, Y)

def gen_model():
    model = Sequential()

    # (((64 - 3) + 0) / 1.) + 1
    model.add(Convolution2D(32, 3, 3, subsample=(1, 1), input_shape=(img_height, img_width, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((62 - 3) + 0) / 1.) + 1
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((60 - 5) + 4) / 1.) + 1
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=0.0001)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def main():

    X_train, Y_train, X_val, Y_val = load_training_and_validation() 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'

    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model()

    filepath = "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

    model.fit_generator(
        transform_generator(X_train, Y_train),
        samples_per_epoch=(len(X_train)*3),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val, is_validation=True),
        nb_val_samples=len(X_val),
        callbacks=[checkpoint])

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    model.save_weights("./outputs/robot.hd5", True)
    with open('./outputs/robot.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == '__main__':
    main()

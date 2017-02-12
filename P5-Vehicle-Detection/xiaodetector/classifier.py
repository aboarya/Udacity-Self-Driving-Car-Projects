import os
import re
import time
import pickle
import glob

import numpy as np
import cv2

from skimage.feature import hog

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class VehicleClassifier(object):
    
    def __init__(self):
        self.clf = clf = Pipeline([('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='hinge')),
               ])

        self.pix_per_cell = 8

        self.orient = 9

        self.cell_per_block = 2

        self.location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def get_hog_features(self, img, feature_vec=True):

        features = hog(img, orientations=self.orient, 
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), 
                       transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec)

        return features
    
    def load_non_vehicle_images(self):
        return glob.glob(self.location+'/../non-vehicles/**/*')
        
    def load_vehicle_images(self):
        return glob.glob(self.location+'/../vehicles/**/*')
    
    def bin_spatial_features(self, img, size):
        # Use cv2.resize().ravel() to create the feature vector
        return cv2.resize(img, size).ravel()
    
    def color_hist_features(self, img, nbins=32, bins_range=(0, 256)):
        
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def extact_image_features(self, image, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):

        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                
        # Apply bin_spatial() to get spatial color features
        spatial_features = self.bin_spatial_features(feature_image, spatial_size)
            
        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist_features(feature_image, nbins=hist_bins)

        ch1 = feature_image[:,:,0]
        ch2 = feature_image[:,:,1]
        ch3 = feature_image[:,:,2]

        hog1 = self.get_hog_features(ch1, feature_vec=False).ravel()

        hog2 = self.get_hog_features(ch2, feature_vec=False).ravel()

        hog3 = self.get_hog_features(ch3, feature_vec=False).ravel()

        hog_features = np.hstack((hog1, hog2, hog3))

        # Append the new feature vector to the features list
        return np.concatenate((spatial_features, hist_features, hog_features))
    
    def extract_features(self, images, cls, spatial_size=(32, 32), hist_bins=32):
        
        # Create a list to append feature vectors to
        features = []
        
        # Iterate through the list of images
        for img in images:
            # Read in each one by one
            image = cv2.imread(img)
            
            features.append(self.extact_image_features(image, hist_bins=hist_bins, spatial_size=spatial_size))

        # Return list of feature vectors and equal length labels
        return (features, [cls] * len(features))
    

    def load_data(self):
    
        print("loading vehicle images")

        vehicle_images = self.load_vehicle_images()
        
        print("load non-vehicle images")

        non_vehicle_images = self.load_non_vehicle_images()
        
        print("extract vehicle features")

        vehicle_features, y_vehicles = self.extract_features(vehicle_images, 1, hist_bins=64, spatial_size=(32, 32))
        
        print("extract non-vehicle features")

        n_vehicle_features, y_n_vehicles = self.extract_features(non_vehicle_images, 0, hist_bins=64, spatial_size=(32, 32))
        
        assert len(vehicle_features) == len(y_vehicles), 'vehicle features and labels are imbalanced'
        
        assert len(n_vehicle_features) == len(y_n_vehicles), 'non vehicle features and labels are imbalanced'
        
        # count = min(len(vehicle_features), len(n_vehicle_features))
        
        # vehicle_features = vehicle_features[:count]
        
        # n_vehicle_features = n_vehicle_features[:count]

        # y_vehicles = y_vehicles[:count]

        # y_n_vehicles = y_n_vehicles[:count]
        
        x = np.vstack((vehicle_features, n_vehicle_features)).astype(np.float64)
        
        y = np.hstack((y_vehicles, y_n_vehicles))

        print("train / test split")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    def save_model(self):
        print("train model")

        t = time.time()
    
        self.clf.fit(self.X_train, self.y_train)

        t2 = time.time()

        print(round(t2-t, 2), 'Seconds to train SVC...')

        with open(self.location+'/model/model.p', 'wb') as _f:
            pickle.dump(self.clf, _f)

        print(self.clf.score(self.X_test, self.y_test))

    def predict(self, features):
        self.clf = pickle.load(open(self.location+'/model/model.p', 'rb'))

        return self.clf.predict(features)

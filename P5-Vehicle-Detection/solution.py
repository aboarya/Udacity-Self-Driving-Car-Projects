import os
import re
import time

import numpy as np
import cv2

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
    
    def _os_walk(self, _dir):
        matches = []
        
        img_re = re.compile(r'.+\.(jpg|png|jpeg|tif|tiff)$', re.IGNORECASE)
        
        for root, dirnames, filenames in os.walk(_dir):
            matches.extend(os.path.join(root, name) for name in filenames if img_re.match(name))
        
        return matches
    
    def load_non_vehicle_images(self):
        return self._os_walk("./non-vehicles")
        
        self.non_vehicle_labels  = [0] * len(non_vehicle_images)
    
    def load_vehicle_images(self):
        return self._os_walk("./vehicles")
    
    def bin_spatial_features(self, img, size):
        # Use cv2.resize().ravel() to create the feature vector
        features = img.ravel()
        
        # Return the feature vector
        return features
    
    def color_hist_features(self, img, nbins=32, bins_range=(0, 256)):
        
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    

    def hog_features(self, img, blockSize=(16, 16), blockStride=(8,8),
                            cellSize=(8,8), winSize=(32, 32), nbins=9,
                            derivAperture=1, winSigma=4., histogramNormType=0,
                            L2HysThreshold=2.0000000000000001e-01,
                            gammaCorrection=0, nlevels=64, winStride=(8,8),
                            padding=(8,8), locations=((10,20),)):
        
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                    derivAperture, winSigma, histogramNormType,
                                    L2HysThreshold, gammaCorrection, nlevels)
            
        #compute(img[, winStride[, padding[, locations]]]) -> descriptors
        return hog.compute(img, winStride, padding, locations)
    
    def extract_features(self, images, cls, cspace='HLS', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
        
        # Create a list to append feature vectors to
        features = []
        
        # Iterate through the list of images
        for file in images:
            # Read in each one by one
            image = cv2.imread(file)
            
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            else: feature_image = np.copy(image)      
                
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial_features(feature_image, spatial_size)
            
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist_features(feature_image, nbins=hist_bins, bins_range=hist_range)
            
            # Apply hog_features() also to get shape related featuers
            hog_features = self.hog_features(feature_image)

            spatial_features = spatial_features.reshape((spatial_features.shape[0], 1))

            hist_features = hist_features.reshape((hist_features.shape[0], 1))

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        
        # Return list of feature vectors and equal length labels
        return (features, [cls] * len(features))
    
    
    
    def load_data(self):
        def shuffle(x, y):
            perm = np.arange(len(x))
            np.random.shuffle(perm)
            x = x[perm]
            y = y[perm]

            return (x, y)
    
        print("loading vehicle images")

        vehicle_images = self.load_vehicle_images()
        
        print("load non-vehicle images")

        non_vehicle_images = self.load_non_vehicle_images()
        
        print("extract vehicle features")

        vehicle_features, y_vehicles = self.extract_features(vehicle_images, 1)
        
        print("extract non-vehicle features")

        n_vehicle_features, y_n_vehicles = self.extract_features(non_vehicle_images, 0)
        
        assert len(vehicle_features) == len(y_vehicles), 'vehicle features and labels are imbalanced'
        
        assert len(n_vehicle_features) == len(y_n_vehicles), 'non vehicle features and labels are imbalanced'
        
        count = min(len(vehicle_features), len(n_vehicle_features))
        
        vehicle_features = vehicle_features[:count]
        
        n_vehicle_features = n_vehicle_features[:count]

        y_vehicles = y_vehicles[:count]

        y_n_vehicles = y_n_vehicles[:count]
        
        x = np.vstack((vehicle_features, n_vehicle_features)).astype(np.float64)
        
        x = x.reshape((x.shape[0], -1), order='F')
        
        y = np.hstack((y_vehicles, y_n_vehicles))

        print("train / test split")
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        
        return (X_train, X_test, y_train, y_test)
    
    def calc_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

def main():
    vc = VehicleClassifier()
    
    X_train, X_test, y_train, y_test = vc.load_data()
    
    # assert len(X) == len(Y), 'imbalanced data'
    
    print("train model")
    t = time.time()
    vc.clf.fit(X_train, y_train)
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    print("testing model")

    # y_pred = vc.clf.predict(X_test)
    
    print('Test Accuracy of classifier = ', round(vlc.clf.score(X_test, y_test), 4))

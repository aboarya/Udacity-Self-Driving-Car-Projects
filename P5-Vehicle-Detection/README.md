
**Vehicle Detection and Tracking Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_feature_scatter_plot.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/test1.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/heatmaps.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

# Overview

* [Introduction](#introduction)
* [Histogram of Oriented Gradients](#histogram-of-oriented-gradients)
* [Vehicle Classification](#vehicle-classification)
* [Sliding Window Search](#sliding-window-search)
* [Video Implementation](#video-implementation)
* [Discussion](#dicussion)

## Introduction

For this project I created a small python package called `xiaodetector` which consists of three modules to carry out each of the required tasks:
 
* `classifier.py` : Uses sklearn's __SVCLinear__ to perform classification
* `detector.py` : Uses OpenCV and a sliding window approach to detect and analyze classifications
* `tracker.py` : Uses an admittadly too simple an algorithm for vehicle tracking

## Histogram of Oriented Gradients

The __VehicleClassifier__ class in the `classifier.py` module carries out the task of training a model using a comination of spatial, color histogram and __HOG__ features.

All images are converted to the __HLS__ color space.  The __Saturation__ channel of the __HLS__ color space is particulary useful for color diffrentiation, while the __Hue__ channel is useful in shape detection when used in __HOG__ feature extaction.

The training of the model is using the Udacity.com provided dataset of Vechiles and Non-Vehicles.

Images are loaded into two separate arrays.  Features for each set of images are extracted in the `extract_features` method of the `classifier.py` module.

Below is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for vehicle and one non-vehicle:


![alt text][image1]

The choice of __HOG__ parameters was a matter of `trial-and-error` by re-training the classifier and running the code against the `test_video.mp4`.  

Inital attemps were made to use an `End-to-Endd` approach whereby no false positives would be detected, however this was found to be not possible.

Flase positives were later elliminated programatically in a combination of the following steps:

* A Heatmap over several frames
* Bounding box width and height validation

## Vehicle Classification

Once feature extaction is complete in the __VehicleClassifier__ module, the data is split into training and test data using `sklearn`'s __train_test_split__.

An __SVCLinear__ classifier in combination with the __StandardScaler__ are used to train the model and the classifier is stored in a pickle in the __models__ directory.

The Classifier was able to score `0.9912` on the test set.

## Sliding Window Search

To improve the efficiency of the efficiency of the detection and tracking, the __HOG__ features at prediction time were taken against an ROI of the image, rather than on independent windows.

The __VechileDetector__ module's `predict` method uses devides the ROI into windows, retrieves a subset of the __HOG__ features, then creates color histogram and spatial features for the window.

All features are then combined and used in classification.  Once a detection is made, a heatmap is updated and threshold of `3` is taken over `5` frames.

After each `5` frames the heatmap is reset.

The __VehicleTracker__ module then makes a decision based on bounding box size of whether the detection is a Vehicle or a false positive.

The __VechileTracker__ module maintians a list of tracked vehicles and updated their positions based on new predictions.

![alt text][image3]

---

## Video Implementation

Example of the processing of the video can be seen int this [link](./output_video.mp4)


Below are all six frames with their bounding boxes and heatmaps

![alt text][image5]


---

## Discussion

There are several issues with my appraoch to this problem.  While the classification and heatmap appraoch are in good standing, the tracking algorith is far too simple.

The tracking is naive and assume any bounding box larger than a specific size is a vehicle.  Tracking does not account for any history of detection and only stored and updated the bounding boxes as the vehicle's positions.

Most notable is overlaping vehicle.  The detection and tracking on overlapping vechicles fails and at times is only able to distinguish both vehicles as one.

The tracking is also not very stable.  The bounding boxes of a detected vehicle are not fixed on the vehicle as it moves.





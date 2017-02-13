import collections

import numpy as np
import cv2

from .detector import VehicleDetector as Detector

class Car(object):

    def __init__(self, position):
        self.position = position
        self.old_position = position

        self.xdiff = [0]
        self.ydiff = [0]

    def compare(self, bbox):
        # print(abs(self.position[0][0] - bbox[0][0]))
        if abs(self.position[0][0] - bbox[0][0]) > 100:
            return True

        self.position = bbox
        self.old_position = self.position
        self.xdiff.append(abs(self.position[0][0] - self.old_position[0][0]))
        self.ydiff.append(abs(self.position[0][1] - self.old_position[0][1]))
        return False

    def draw(self, img):

        w = 150
        h = 75

        x1, y1, x2, y2 = self.position[0][0], self.position[0][1], self.position[0][0]+w, self.position[0][1]+h

        cv2.rectangle(img, (x1+sum(self.xdiff), y1+sum(self.ydiff)), (x2, y2), (0,0,255), 6)


class VehicleTracker(object):

    def __init__(self):
        self.detector = Detector()

        self.cars = collections.deque()

    def is_car(self, bbox):
        car = True
        # check the width and height of the box
        # print(abs(bbox[0][0] - bbox[1][0]))
        # print(abs(bbox[0][1] - bbox[1][1]))
        # print(">>>>>>>>>>>>>>")
        if abs(bbox[0][0] - bbox[1][0]) < 75 or abs(bbox[0][1] - bbox[1][1]) < 75:
            car = False

        return car

    def update_cars(self, bbox):

        if len(self.cars) == 0:
            self.cars.append(Car(bbox))

        new = False
        for car in list(self.cars):
            if car.compare(bbox):
                # print("is new car")
                new = True

        if new:
            self.cars.append(Car(bbox))

    def process_labels(self, labels):
        # Iterate through all detected cars

        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
        
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            
            nonzerox = np.array(nonzero[1])
        
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            if self.is_car(bbox):
                self.update_cars(bbox)

    def draw_cars(self, img):

        for car in self.cars:
            car.draw(img)


    def track(self, image):
        labels = self.detector.detect(image)

        self.process_labels(labels)

        im = np.copy(image)

        self.draw_cars(im)

        return im
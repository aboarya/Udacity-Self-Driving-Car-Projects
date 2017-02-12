import cv2
import collections

import numpy as np

from scipy.ndimage.measurements import label

from .classifier import VehicleClassifier as Classifier

from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def inOtherRect(rect_inner,rect_outer):
    return rect_inner[0]>=rect_outer[0] and \
        rect_inner[0]+rect_inner[2]<=rect_outer[0]+rect_outer[2] and \
        rect_inner[1]>=rect_outer[1] and \
        rect_inner[1]+rect_inner[3]<=rect_outer[1]+rect_outer[3] and \
        (rect_inner!=rect_outer)

def centroid_function(detections, img, heat_map):
    
    centroid_rectangles = []
    
    # _, binary = cv2.threshold(heat_map, 11, 255, cv2.THRESH_BINARY);

    _, contours, _ = cv2.findContours(heat_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    outer_rects =[ cv2.boundingRect(c) for c in contours[:]]
    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        for contour1 in contours:
            rectangle1 = cv2.boundingRect(contour1)
            if inOtherRect(rectangle, rectangle1):
                if rectangle in outer_rects:
                    outer_rects.remove(rectangle)

        # rectangle = cv2.boundingRect(contour)
        # if rectangle[2] < 40 or rectangle[3] < 40: continue
        # x,y,w,h = rectangle
        # centroid_rectangles.append([x,y,x+w,y+h])

    for rectangle in outer_rects:
        # rectangle = cv2.boundingRect(contour)
        if rectangle[2] < 50 or rectangle[3] < 50: continue
        x,y,w,h = rectangle
        centroid_rectangles.append([x,y,x+w,y+h])



    return centroid_rectangles

class Object:
    def __init__(self,position):
        self.position = position
        self.new_postion = None
        self.count = 0
        self.frame = 1
        self.flag = False
        self.long_count = 0
        self.postion_average = []

    def update(self,temp_position):
        if abs(temp_position[2]-self.position[2]) < 100 and abs(temp_position[3]-self.position[3]) < 100:
            if self.long_count > 2:
                self.postion_average.pop(0)
                self.postion_average.append(temp_position)
                self.new_postion = np.mean(np.array(self.postion_average), axis=0).astype(int)
                self.position = self.new_postion
                self.frame = 1
                self.count += 1

                return False

            self.position = temp_position
            self.postion_average.append(temp_position)
            self.count+=1

            return False

        else:
            return True

    def get_position(self):
        self.frame+=1
        if self.count == 7 and self.long_count < 3 :
            self.new_postion = np.mean(np.array(self.postion_average), axis=0).astype(int)
            self.count = 0
            self.frame = 1
            self.long_count += 1
            if self.long_count < 2:
                self.postion_average = []

        if self.frame > 10:
            self.flag = True

        return self.new_postion, self.flag

class Vehicle(Object):
     def __init__(self, position):
        Object.__init__(self, position)

class Person(Object):
    def __init__(self, position):
        Object.__init__(self, position)

class VehicleDetector(object):

    def __init__(self):
        self.classifier = Classifier()

        self.count = 1

        self.cars = list()

        self.boxes = collections.deque()

        self.heatmap = np.zeros((720, 1280), np.uint8)

    def update_heatmap(self, candidates, image_shape, heatmap=None):
        if heatmap is None:
            heatmap = np.zeros((image_shape[0], image_shape[1]), np.uint8)

        for pt1, pt2 in candidates:
            x1, y1 = pt1
        
            x2, y2 = pt2
            
            x1 = min(max(x1, 0), image_shape[1])
            
            x2 = min(max(x2, 0), image_shape[1])
            
            y1 = min(max(y1, 0), image_shape[0])
            
            y2 = min(max(y2, 0), image_shape[0])
            
            xv, yv = np.meshgrid(range(x1, x2), range(y1, y2))

            heatmap[yv, xv] += 1

        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
        
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            
            nonzerox = np.array(nonzero[1])
        
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # bbox, flag  = self.is_new_car(bbox)

            self.boxes.append((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
        
            # Draw the box on the image
            # cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

            # if flag: self.boxes.append(bbox)
        
        # Return the image
        return img

    def detect(self, image):

        scale = 1.5

        y_start_stop = [400, 656] # Min and max in y to search in slide_window()

        draw_image = np.copy(image)

        # windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
        #             xy_window=(64, 64), xy_overlap=(0.5, 0.5))

        roi_window = draw_image[y_start_stop[0]:y_start_stop[1],:,:]

        roi_window = cv2.resize(roi_window, (np.int(roi_window.shape[1]/scale), np.int(roi_window.shape[0]/scale)))

        feature_image = cv2.cvtColor(roi_window, cv2.COLOR_BGR2YCrCb)

        ch1 = feature_image[:,:,0]
        ch2 = feature_image[:,:,1]
        ch3 = feature_image[:,:,2]

        hog1 = get_hog_features(ch1, 9, 8, 2, feature_vec=False)

        hog2 = get_hog_features(ch2, 9, 8, 2, feature_vec=False)

        hog3 = get_hog_features(ch3, 9, 8, 2, feature_vec=False)

        orient = 9  # HOG orientations
        
        pix_per_cell = 8 # HOG pixels per cell
        
        cell_per_block = 2 # HOG cells per block
        
        window = 64

        nxblocks = (roi_window.shape[1] // pix_per_cell) - 1
        
        nyblocks = (roi_window.shape[0] // pix_per_cell) - 1
        
        nfeat_per_block = orient*cell_per_block**2

        nblocks_per_window = (window // pix_per_cell) -1
        
        cells_per_step = 2

        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step

        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        if self.count %  3 == 0:
            self.heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        # self.heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        # self.heatmap //= self.count
        self.boxes.clear()

        for xb in range(nxsteps):
            for yb in range(nysteps):

                try:
                    ypos = yb*cells_per_step

                    xpos = xb*cells_per_step

                    hog1_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    
                    hog2_features = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    
                    hog3_features = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                    hog_features = np.hstack((hog1_features, hog2_features, hog3_features))

                    xleft = xpos*pix_per_cell

                    ytop = ypos*pix_per_cell

                    subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64, 64))

                    spatial_features = self.classifier.bin_spatial_features(subimg, (32, 32))

                    color_features = self.classifier.color_hist_features(subimg)

                    test_features = np.hstack((spatial_features, color_features, hog_features)).reshape((1, -1))
                
                    prediction = self.classifier.predict(test_features)

                    if prediction == 1:
                        xbox_left = np.int(xleft*scale)

                        ytop_draw = np.int(ytop*scale)

                        win_draw = np.int(window*scale)

                        self.boxes.append((xbox_left, ytop_draw+y_start_stop[0], xbox_left+win_draw, ytop_draw+win_draw+y_start_stop[0]))

                        self.heatmap[ytop_draw+y_start_stop[0]:ytop_draw+win_draw+y_start_stop[0], xbox_left:xbox_left+win_draw] += 1

                except Exception as e:
                    raise e
    
        self.count += 1

        self.heatmap[self.heatmap < 3] = 0
        
        # cv2.GaussianBlur(self.heatmap, (31, 31), 0, dst=self.heatmap)
        
        # labels = label(self.heatmap)

        # print(labels[1])
    
        # im = self.draw_labeled_bboxes(np.copy(image), labels)

        im = np.copy(image)

        centroids = centroid_function(self.boxes, im, self.heatmap)

        for centroid in centroids:
            new = True
            for car in self.cars:
                new = car.update(centroid)
            if new == False:
                break
            if new == True:
                self.cars.append(Vehicle(centroid))

        next_cars = []
        positions = []

        for car in self.cars:
            position, flag = car.get_position()
            if flag == False:
                next_cars.append(car)
                positions.append(position)

        self.cars = next_cars

        try:
            for (x1, y1, x2, y2) in positions:
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        except:
            pass

        # im = np.copy(image)

        # for centroid in self.centroids:
        #     new = True
        #     for car in self.cars:
        #         new = car.update(centroid)
        #         if new == False:
        #             break
        #     if new == True:
        #         self.cars.append(Vehicle(centroid))

        # next_cars = []
        # positions = []

        # for car in self.cars:
        #     position, flag = car.get_position()
        #     if flag == False:
        #         next_cars.append(car)
        #     positions.append(position)

        # self.cars = next_cars

        # try:
        #     for (x1, y1, x2, y2) in positions:
        #         cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        # except:
        #     pass

        return im

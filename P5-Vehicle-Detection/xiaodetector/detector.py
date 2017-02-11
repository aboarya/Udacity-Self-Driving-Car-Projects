import cv2
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

class VehicleDetector(object):

    def __init__(self):
        self.classifier = Classifier()

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
        
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
        # Return the image
        return img

    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
    
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
    
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
    
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
    
        yspan = y_start_stop[1] - y_start_stop[0]
    
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
    
        # Initialize a list to append window positions to
        window_list = []
    
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                
                endx = startx + xy_window[0]
                
                starty = ys*ny_pix_per_step + y_start_stop[0]
                
                endy = starty + xy_window[1]
            
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        
        # Return the list of windows
        return window_list

    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            
            #4) Extract features for that window using single_img_features()
            features = self.classifier.extact_image_features(test_img, cspace='YUV', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
        
            #5) Scale extracted features to be fed to classifier

            features = features.astype(np.float64)
        
            #6) Predict using your classifier
            prediction = self.classifier.predict(features)
        
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
    
        #8) Return windows for positive detections
        return on_windows

    def detect(self, image):
        # Define a function to draw bounding boxes
        def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
            # Make a copy of the image
            imcopy = np.copy(img)
            # Iterate through the bounding boxes
            for bbox in bboxes:
                # Draw a rectangle given bbox coordinates
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
                # Return the image copy with boxes drawn
            return imcopy

        y_start_stop = [400, 656] # Min and max in y to search in slide_window()

        draw_image = np.copy(image)

        # windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
        #             xy_window=(64, 64), xy_overlap=(0.5, 0.5))

        roi_window = draw_image[y_start_stop[0]:y_start_stop[1],:,:]

        roi_window = cv2.resize(roi_window, (np.int(roi_window.shape[1]/1.5), np.int(roi_window.shape[0]/1.5)))

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

        count = 0

        scale = 1.5
        
        heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        for xb in range(nxsteps):
            for yb in range(nysteps):

                count += 1

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

                        cv2.rectangle(draw_image, (xbox_left, ytop_draw+y_start_stop[0]), (xbox_left+win_draw, ytop_draw+win_draw+y_start_stop[0]),  (0,0,255), 6)

                        heatmap[ytop_draw+y_start_stop[0]:ytop_draw+win_draw+y_start_stop[0], xbox_left:xbox_left+win_draw] += 1 

                except Exception as e:
                    raise e
    
        heatmap[heatmap < 2] = 0
        
        cv2.GaussianBlur(heatmap, (31, 31), 0, dst=heatmap)
        
        labels = label(heatmap)
    
        im = self.draw_labeled_bboxes(np.copy(image), labels)

        return im, labels

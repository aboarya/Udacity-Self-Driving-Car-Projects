

class VehicleDetector(object):

	def __init__(self):
		self.clf = Classifer()

	def centroid(box, as_int=False):
    	x1, y1 = box[0]
    	
    	x2, y2 = box[1]
    	
    	if not as_int:
        	return ((x1+x2)/2., (y1+y2)/2.)
    	
    	else:
        	return (int((x1+x2)//2), ((y1+y2)//2))

    def update_heatmap(candidates, image_shape, heatmap=None):
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


    def draw_labeled_bboxes(img, labels):
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
	def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
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
	def search_windows(img, windows, vc):

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

    def detect(self):
    	y_start_stop = [350, 720] # Min and max in y to search in slide_window()

		image = cv2.imread('test_images/test5.jpg')

		draw_image = np.copy(image)

		windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))

		hot_windows = self.search_windows(image, windows, vc)

		centroids = np.array([centroid(w) for w in hot_windows])

		current_heatmap = update_heatmap(hot_windows, (720, 1280))

		thresh_heatmap = current_heatmap
	
		thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
		
		cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)
		
		labels = label(thresh_heatmap)
	
		im2 = draw_labeled_bboxes(np.copy(image), labels)
		
		plt.imshow(im2)


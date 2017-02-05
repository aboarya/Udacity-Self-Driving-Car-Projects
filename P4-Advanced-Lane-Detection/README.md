**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undisorted_frame0.jpg "Undistorted"
[image1]: ./output_images/undistored_road_frame0.jpg "Road Transformed"
[image3]: ./output_images/binary_example_frame0.jpg "Binary Example"
[image4]: ./output_images/warp_example_frame0.jpg "Warp Example"
[image5]: ./output_images/fit_visual_frame0.jpg "Fit Visual"
[image6]: ./output_images/result_frame0.jpg "Output"
[video7]: ./output_video.mp4 "Video"

# Overview

* [Introduction](#introduction)
* [Camera Calibration](#camera-calibration)
* [Pipeline](#pipeline)

## Introduction

The code for this project is structured in `solution.py` in this directory.  

The solution consists of two classes.  The ___LaneDetection___, and the ___Line___ class.

The ___Line___ class is simply a data class while the ___LaneDetection___ class contains the workflow required for this project.

Exporting the video is done using `moviepy`'s ___VideoFileClip___ API, which processes a video one frame at a time.  The __LaneDetection__'s `process_image` method is give to the API to create `output_video.mp4`.


## Camera Calibration

The constructor of the ___LaneDetection___ class accepts a set of calibration images as well as a test image on which to test the calibration.

The constructor calls the `calibrate_camera` method, which creates one set of *object points* and one set of *images points* for each of the calibration images; which are images of a chessboard from different perspectives.  

The *image points* are the corners of the black and white squares of the chessboard which are dicoverable via __cv2__'s API, while the *object points* are the coordinates of these squares within the image in `(x , y, z)`.

The image and object points create an `undistortion` matrix which is applied to the below image.


![alt text][image1]

## Pipeline (single images)

The pipeline for this project can best be summed up as follows:

* Undistort Image
* Binary Threshold Image
* Perspective Transform Image
* Detect lane lines using sliding windows and mark as detected
* Project lane onto input image

This pipeline is found in the code in the `process_image` method as follows:


```
binary = ld.binary_transform(ld.undistort(
            img), thresh_min=40, thresh_max=200, ksize=3, thresh=(0.7, 1.3), hls_thresh=(175, 255))

binary_warped = ld.perspective_transform(binary)

# Generate x and y values for plotting
self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

if self.left_line.detected and self.right_line.detected:
    return self.detect_lines_from_previous(binary_warped, img)

return self.detect_lines_using_windows(binary_warped, img)
```
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

####Undistort Image
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####Binary Threshold Image

Binay thresholding is done in the `binary_transform` method of the __LaneDetection__ class and is done in sequence of the following

* Gradient Thresholding
* Color Thresholding
* Combined Gradient and Color

Each of the thresholding steps is done in their respective methods which are called within the `binary_transform` method.

The `gradient thresholding` is done in 4 steps:

* Thresholding over the X axis
* Thresholding over the Y axis
* Thresholing of magnitude
* Thresholding over direction
* Combining the above

####Perspective Transform

The code for my perspective transform includes a function called `perspective_transform` method of the __LaneDetection__, I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

The output of which can be seen below

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


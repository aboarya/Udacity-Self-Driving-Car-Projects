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
[image2]: ./output_images/undistored_road_frame0.jpg "Road Transformed"
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

Exporting the video is done using `moviepy`'s ___VideoFileClip___ API, which processes a video one frame at a time.  The __LaneDetection__'s `process_image` method is given to the API to create `output_video.mp4`.


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


####Undistort Image

The `undistortion` is applied to each frame and an example is seen below.

![alt text][image2]

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

The output of the above can be seen in this image:

![alt text][image3]

####Perspective Transform

The code for my perspective transform includes a function called `perspective_transform` in the __LaneDetection__, I chose the hardcode the source and destination points in the following manner:

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

####Lane Detection

Once a threasholded image has been warped, the curvature of the lanes can easily be detected.

The first step in this process is applying a histogram to the bottom half of the warped binary image.

Since the binary image is simply zeros and ones, the location of the lanes in a historgam is clearly identifiable by the peaks of the histogram.  

The histogram represents all the `1`'s in the binary image and the lane are a peak within that representation.

Using only the bottom half of the image provides us with a place to begin the search for the entire lane.  

We can use a small window from the bottom of the peak of the histogram to begin the search.

The search within the window is similar to the idea of the histogram approach; a concentration of `ones` are the lane and `zeors` are everything else that was removed by our binary thresholding 

A small size for the window is best.  I choose 9 windows of 100 pixels.  The indices of the detected `ones` are added to a list, and the windows `slides` up towards the top of the image.

We can safely assume the camera is in the center of the car, therefore the right half of the binary image is where the right lane would be and the left one accordingly.

Once we have a group of indices seperated by location; left and right, we can use the __cv2__ API to create a second order polynomial to represent the curvature of the lane.

This gives us a set of coefficeints similar to `aX**2 + bX + c`.  We can use these coefficients to calculte the radius of the curvature of the lane for one of several reasons:

* Sanity check to ensure we detected a proper lane segment
* Storing curvature to sanity check against future sections of the road where light conditions may not be ideal for thresholding.

The above workflow is contained in the `detect_lines_using_windows` method of the __LaneDetection__ class and an example of its output is below

![alt text][image5]

#### Radius Calculation

Radius calculation; in order to be useful for sanity check, must be converted to meters.

There's an excellent tutorial and sample code in the lesson on how to do this.

Mostly importantly is convertion from pixel space to real world (meter space).

This was done in the `radius_to_meters` of the __LaneDetection__ class by reapplying the ploynimials with a conversion to `real world` space.  

Once that is done, the `aX**2 + bX + c` formula is reapplied to calculate the curvature in meters.

#### Example Ouput

The above workflow is contained in the `process_image` method and an ouput image is seen below.

![alt text][image6]

---

###Pipeline (video)



A complete video example [link to my video result](./output_video.mp4)

---

###Discussion

Although this piple works reasonably well on the project video, it is likely to fail under challenging road segments.

This is mainly caused by the fact that binary thresholding under poor light conditions, shade, and bad weather conditions is likely to produce multiple faint histograms that make the lane detection difficult.

As mentioned, the radius can be used for sanity check and a better algorithm can be used such that the if detection doesn't make sense, the previous frame can be used instead since the road is likely to be continous.

This can be done up to a certain count of frames before resuming the detection once again.




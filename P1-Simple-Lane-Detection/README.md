
# **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  


```python
#importing some useful packages
import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import numpy as np
import cv2

%matplotlib inline
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x10b281438>




![png](output_4_2.png)


**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import numpy as np
import cv2

def load_image(image_path):
    """ pass for now """
    #return (mpimg.imread(image_path)*255).astype('uint8')
    return mpimg.imread(image_path)


def save_plot(dir_name, name, img):
    """ save plot to current directory """
    cv2.imwrite(join(dir_name, name), img)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_noise(img, kernel_size):
    """ Applies a Gaussian Noise kernel """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """ Applies the Canny transform """
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = (255,)

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255,0,0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    right_slope = []
    left_slope = []

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y1-y2) / (x1-x2)) # slope
            if m <= -0.2:
                left_slope.append(m)
                left_lines.append((x1, y1))
            elif m >= 0.2 and m <= 0.88:
                right_slope.append(m)
                right_lines.append((x2, y2))

    # average left and right slopes
    right_slope = np.mean(right_slope)
    left_slope = np.mean(left_slope)

    # x = np.array([x[0] for x in right_lines])
    # y = np.array([x[1] for x in right_lines])
    # raise RuntimeError(np.polyfit(x, y, 3))
    
    # start_left_y = sorted([line[1] for line in left_lines])[int(len(left_lines)/2)]
    # start_left_x = [line[0] for line in left_lines if line[1] == start_left_y][0]

    # start_right_y = sorted([line[1] for line in right_lines])[int(len(right_lines)/2)]
    # start_right_x = [line[0] for line in right_lines if line[1] == start_right_y][0]

    min_right_x1 = sorted(right_lines, key=lambda x: x[1])[0][0]
    min_right_y1 = sorted(left_lines, key=lambda x: x[1])[0][1]

    min_left_x1 = sorted(left_lines, key=lambda x: x[1])[0][0]
    min_left_y1 = sorted(left_lines, key=lambda x: x[1])[0][1]
    
    # x2 = ((y2-y1)/m) + x1 where y2 = max height
    # first we pick a start point on the horizon
    
    start_left_y = start_right_y = 325 # point on horizon
    start_right_x = int((start_right_y - min_right_y1) / right_slope) + min_right_x1
    start_left_x = int((start_right_y - min_left_y1) / left_slope) + min_left_x1

    # next we extend to the car
    end_left_x = int((img.shape[1]-start_left_y)/left_slope) + start_left_x
    end_right_x = int((img.shape[1]-start_right_y)/right_slope) + start_right_x
    
    cv2.line(img, (start_left_x, start_left_y), (end_left_x, img.shape[1]), color, thickness)
    cv2.line(img, (start_right_x, start_right_y), (end_right_x, img.shape[1]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    #for line in lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(line_img, (x1, y1), (x2, y2), [255,0,0], 2)
    return line_img


def weighted_img(img, initial_img, alpha=0.7, beta=0.5, upsilon=0.3):
    """
    Python 3 has support for cool math symbols.
    `img` is the output of the hough_lines(), An image with lines drawn on it
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + upsilon 
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, upsilon)


def parse_args():
    """ pass me """
    parser = argparse.ArgumentParser()
    parser.add_argument('test_on',
                        choices=('images', 'videos',),
                        help='Is this an image or a video test?')
    parser.add_argument('dir_path',
                        help='Path to directory containing images / videos')
    return parser.parse_args()

def get_files(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def process_image(img):

    initial_image = np.copy(img)
    gray_img = grayscale(img)
    blur_gray = gaussian_noise(gray_img, 3)

    edges = canny(blur_gray, 40, 70) #31
    
    imshape = img.shape

    vertices = np.array([[(105, .888*imshape[0]),
        (.333*imshape[1],
            .708*imshape[0]),
        (.528*imshape[1],
            .597*imshape[0]),
        (imshape[1], .805*imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    lines = hough_lines(masked_edges, 1, np.pi/180, 25, 10, 10)
    
    zeros = np.zeros_like(lines)
    lines = np.dstack((lines, zeros, zeros))
    final_img = weighted_img(lines, initial_image)
    return final_img

```

## Test on Images

Now you should build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidWhiteCurve.jpg',
     'solidWhiteRight.jpg',
     'solidYellowCurve.jpg',
     'solidYellowCurve2.jpg',
     'solidYellowLeft.jpg',
     'whiteCarLaneSwitch.jpg']



run your solution on all test_images and make copies into the test_images directory).


```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

images = get_files('./test_images')

for name in images:

    if name.startswith("."): continue
        
    print("processing", name)
    
    img = load_image('{}/{}'.format('./test_images', name))
    
    img = process_image(img)
    
    plt.imshow(img)
```

    processing solidWhiteCurve.jpg
    processing solidWhiteRight.jpg
    processing solidYellowCurve.jpg
    processing solidYellowCurve2.jpg
    processing solidYellowLeft.jpg
    processing whiteCarLaneSwitch.jpg



![png](output_11_1.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_images(name, img):
    """
    # 1: grayscale the image
    # 2: define edges
    # 3: Hough transform
    # 4: Apply ROI
    """
    final_img = processs_image(img)


def process_video(name):
    """ ok """
    _output = 'final_'+name
    clip1 = VideoFileClip(name)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(_output, audio=False)
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'white.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    
    
    
    
    
      0%|          | 0/222 [00:00<?, ?it/s][A[A[A[A[A
    
    
    
    
      4%|▎         | 8/222 [00:00<00:02, 76.38it/s][A[A[A[A[A
    
    
    
    
      9%|▊         | 19/222 [00:00<00:02, 83.40it/s][A[A[A[A[A
    
    
    
    
     14%|█▎        | 30/222 [00:00<00:02, 88.71it/s][A[A[A[A[A
    
    
    
    
     18%|█▊        | 41/222 [00:00<00:01, 93.43it/s][A[A[A[A[A
    
    
    
    
     22%|██▏       | 49/222 [00:00<00:02, 78.45it/s][A[A[A[A[A
    
    
    
    
     26%|██▌       | 57/222 [00:00<00:02, 76.79it/s][A[A[A[A[A
    
    
    
    
     29%|██▉       | 65/222 [00:00<00:02, 74.65it/s][A[A[A[A[A
    
    
    
    
     33%|███▎      | 73/222 [00:00<00:02, 66.80it/s][A[A[A[A[A
    
    
    
    
     36%|███▌      | 80/222 [00:01<00:02, 61.93it/s][A[A[A[A[A
    
    
    
    
     39%|███▉      | 87/222 [00:01<00:02, 58.95it/s][A[A[A[A[A
    
    
    
    
     42%|████▏     | 93/222 [00:01<00:02, 55.89it/s][A[A[A[A[A
    
    
    
    
     45%|████▍     | 99/222 [00:01<00:02, 53.99it/s][A[A[A[A[A
    
    
    
    
     47%|████▋     | 105/222 [00:01<00:02, 53.91it/s][A[A[A[A[A
    
    
    
    
     50%|█████     | 111/222 [00:01<00:02, 55.03it/s][A[A[A[A[A
    
    
    
    
     53%|█████▎    | 117/222 [00:01<00:01, 53.69it/s][A[A[A[A[A
    
    
    
    
     55%|█████▌    | 123/222 [00:01<00:01, 54.91it/s][A[A[A[A[A
    
    
    
    
     58%|█████▊    | 129/222 [00:01<00:01, 53.58it/s][A[A[A[A[A
    
    
    
    
     61%|██████▏   | 136/222 [00:02<00:01, 55.46it/s][A[A[A[A[A
    
    
    
    
     64%|██████▍   | 142/222 [00:02<00:01, 54.45it/s][A[A[A[A[A
    
    
    
    
     67%|██████▋   | 148/222 [00:02<00:01, 50.01it/s][A[A[A[A[A
    
    
    
    
     69%|██████▉   | 154/222 [00:02<00:01, 50.33it/s][A[A[A[A[A
    
    
    
    
     73%|███████▎  | 162/222 [00:02<00:01, 55.24it/s][A[A[A[A[A
    
    
    
    
     76%|███████▌  | 169/222 [00:02<00:00, 58.68it/s][A[A[A[A[A
    
    
    
    
     79%|███████▉  | 176/222 [00:02<00:00, 61.61it/s][A[A[A[A[A
    
    
    
    
     83%|████████▎ | 185/222 [00:02<00:00, 67.36it/s][A[A[A[A[A
    
    
    
    
     87%|████████▋ | 194/222 [00:03<00:00, 72.28it/s][A[A[A[A[A
    
    
    
    
     91%|█████████ | 202/222 [00:03<00:00, 72.99it/s][A[A[A[A[A
    
    
    
    
     95%|█████████▌| 211/222 [00:03<00:00, 74.71it/s][A[A[A[A[A
    
    
    
    
     99%|█████████▉| 220/222 [00:03<00:00, 77.89it/s][A[A[A[A[A
    
    
    
    
    100%|█████████▉| 221/222 [00:03<00:00, 66.09it/s][A[A[A[A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4 
    
    CPU times: user 2.59 s, sys: 722 ms, total: 3.32 s
    Wall time: 3.8 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('./test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    
    
    
    
    
      0%|          | 0/682 [00:00<?, ?it/s][A[A[A[A[A
    
    
    
    
      1%|▏         | 9/682 [00:00<00:07, 85.87it/s][A[A[A[A[A
    
    
    
    
      3%|▎         | 20/682 [00:00<00:07, 90.37it/s][A[A[A[A[A
    
    
    
    
      4%|▍         | 30/682 [00:00<00:07, 92.95it/s][A[A[A[A[A
    
    
    
    
      6%|▌         | 40/682 [00:00<00:06, 94.60it/s][A[A[A[A[A
    
    
    
    
      7%|▋         | 48/682 [00:00<00:08, 77.86it/s][A[A[A[A[A
    
    
    
    
      8%|▊         | 55/682 [00:00<00:09, 69.34it/s][A[A[A[A[A
    
    
    
    
      9%|▉         | 62/682 [00:00<00:09, 66.15it/s][A[A[A[A[A
    
    
    
    
     10%|█         | 69/682 [00:00<00:09, 62.98it/s][A[A[A[A[A
    
    
    
    
     11%|█         | 76/682 [00:01<00:10, 59.69it/s][A[A[A[A[A
    
    
    
    
     12%|█▏        | 83/682 [00:01<00:10, 59.67it/s][A[A[A[A[A
    
    
    
    
     13%|█▎        | 89/682 [00:01<00:10, 57.70it/s][A[A[A[A[A
    
    
    
    
     14%|█▍        | 97/682 [00:01<00:09, 61.19it/s][A[A[A[A[A
    
    
    
    
     15%|█▌        | 104/682 [00:01<00:09, 62.14it/s][A[A[A[A[A
    
    
    
    
     16%|█▋        | 111/682 [00:01<00:09, 61.30it/s][A[A[A[A[A
    
    
    
    
     17%|█▋        | 118/682 [00:01<00:09, 59.31it/s][A[A[A[A[A
    
    
    
    
     18%|█▊        | 124/682 [00:01<00:10, 54.62it/s][A[A[A[A[A
    
    
    
    
     19%|█▉        | 130/682 [00:01<00:10, 54.73it/s][A[A[A[A[A
    
    
    
    
     20%|█▉        | 136/682 [00:02<00:10, 52.88it/s][A[A[A[A[A
    
    
    
    
     21%|██        | 142/682 [00:02<00:10, 50.21it/s][A[A[A[A[A
    
    
    
    
     22%|██▏       | 148/682 [00:02<00:10, 49.40it/s][A[A[A[A[A
    
    
    
    
     23%|██▎       | 154/682 [00:02<00:10, 50.86it/s][A[A[A[A[A
    
    
    
    
     24%|██▎       | 161/682 [00:02<00:09, 54.60it/s][A[A[A[A[A
    
    
    
    
     25%|██▍       | 168/682 [00:02<00:08, 57.32it/s][A[A[A[A[A
    
    
    
    
     26%|██▌       | 175/682 [00:02<00:08, 59.09it/s][A[A[A[A[A
    
    
    
    
     27%|██▋       | 184/682 [00:02<00:07, 64.57it/s][A[A[A[A[A
    
    
    
    
     28%|██▊       | 191/682 [00:03<00:07, 65.57it/s][A[A[A[A[A
    
    
    
    
     29%|██▉       | 198/682 [00:03<00:07, 62.18it/s][A[A[A[A[A
    
    
    
    
     30%|███       | 205/682 [00:03<00:07, 61.23it/s][A[A[A[A[A
    
    
    
    
     31%|███       | 213/682 [00:03<00:07, 65.57it/s][A[A[A[A[A
    
    
    
    
     32%|███▏      | 221/682 [00:03<00:06, 67.97it/s][A[A[A[A[A
    
    
    
    
     33%|███▎      | 228/682 [00:03<00:07, 61.36it/s][A[A[A[A[A
    
    
    
    
     34%|███▍      | 235/682 [00:03<00:07, 61.30it/s][A[A[A[A[A
    
    
    
    
     35%|███▌      | 242/682 [00:03<00:07, 58.68it/s][A[A[A[A[A
    
    
    
    
     36%|███▋      | 248/682 [00:03<00:07, 56.08it/s][A[A[A[A[A
    
    
    
    
     37%|███▋      | 254/682 [00:04<00:07, 56.79it/s][A[A[A[A[A
    
    
    
    
     38%|███▊      | 260/682 [00:04<00:07, 56.79it/s][A[A[A[A[A
    
    
    
    
     39%|███▉      | 268/682 [00:04<00:06, 61.45it/s][A[A[A[A[A
    
    
    
    
     41%|████      | 277/682 [00:04<00:06, 66.74it/s][A[A[A[A[A
    
    
    
    
     42%|████▏     | 285/682 [00:04<00:05, 69.59it/s][A[A[A[A[A
    
    
    
    
     43%|████▎     | 293/682 [00:04<00:05, 71.00it/s][A[A[A[A[A
    
    
    
    
     44%|████▍     | 301/682 [00:04<00:05, 71.89it/s][A[A[A[A[A
    
    
    
    
     45%|████▌     | 309/682 [00:04<00:05, 67.87it/s][A[A[A[A[A
    
    
    
    
     46%|████▋     | 316/682 [00:04<00:05, 68.01it/s][A[A[A[A[A
    
    
    
    
     48%|████▊     | 324/682 [00:05<00:05, 69.44it/s][A[A[A[A[A
    
    
    
    
     49%|████▊     | 332/682 [00:05<00:05, 67.01it/s][A[A[A[A[A
    
    
    
    
     50%|████▉     | 339/682 [00:05<00:05, 66.66it/s][A[A[A[A[A
    
    
    
    
     51%|█████     | 346/682 [00:05<00:04, 67.56it/s][A[A[A[A[A
    
    
    
    
     52%|█████▏    | 354/682 [00:05<00:04, 69.35it/s][A[A[A[A[A
    
    
    
    
     53%|█████▎    | 362/682 [00:05<00:04, 71.18it/s][A[A[A[A[A
    
    
    
    
     54%|█████▍    | 370/682 [00:05<00:04, 70.13it/s][A[A[A[A[A
    
    
    
    
     55%|█████▌    | 378/682 [00:05<00:04, 70.67it/s][A[A[A[A[A
    
    
    
    
     57%|█████▋    | 386/682 [00:05<00:04, 72.04it/s][A[A[A[A[A
    
    
    
    
     58%|█████▊    | 394/682 [00:06<00:03, 72.90it/s][A[A[A[A[A
    
    
    
    
     59%|█████▉    | 402/682 [00:06<00:03, 70.15it/s][A[A[A[A[A
    
    
    
    
     60%|██████    | 410/682 [00:06<00:03, 69.38it/s][A[A[A[A[A
    
    
    
    
     61%|██████    | 417/682 [00:06<00:04, 63.07it/s][A[A[A[A[A
    
    
    
    
     62%|██████▏   | 424/682 [00:06<00:04, 60.43it/s][A[A[A[A[A
    
    
    
    
     63%|██████▎   | 432/682 [00:06<00:03, 63.96it/s][A[A[A[A[A
    
    
    
    
     64%|██████▍   | 439/682 [00:06<00:03, 65.47it/s][A[A[A[A[A
    
    
    
    
     65%|██████▌   | 446/682 [00:06<00:03, 64.98it/s][A[A[A[A[A
    
    
    
    
     66%|██████▋   | 453/682 [00:07<00:03, 61.34it/s][A[A[A[A[A
    
    
    
    
     67%|██████▋   | 460/682 [00:07<00:03, 61.64it/s][A[A[A[A[A
    
    
    
    
     68%|██████▊   | 467/682 [00:07<00:03, 58.43it/s][A[A[A[A[A
    
    
    
    
     70%|██████▉   | 474/682 [00:07<00:03, 61.25it/s][A[A[A[A[A
    
    
    
    
     71%|███████   | 482/682 [00:07<00:03, 64.36it/s][A[A[A[A[A
    
    
    
    
     72%|███████▏  | 490/682 [00:07<00:02, 66.42it/s][A[A[A[A[A
    
    
    
    
     73%|███████▎  | 497/682 [00:07<00:03, 60.73it/s][A[A[A[A[A
    
    
    
    
     74%|███████▍  | 504/682 [00:07<00:02, 60.60it/s][A[A[A[A[A
    
    
    
    
     75%|███████▍  | 511/682 [00:07<00:02, 63.01it/s][A[A[A[A[A
    
    
    
    
     76%|███████▌  | 518/682 [00:08<00:02, 61.62it/s][A[A[A[A[A
    
    
    
    
     77%|███████▋  | 525/682 [00:08<00:02, 55.87it/s][A[A[A[A[A
    
    
    
    
     78%|███████▊  | 531/682 [00:08<00:02, 56.06it/s][A[A[A[A[A
    
    
    
    
     79%|███████▊  | 537/682 [00:08<00:02, 54.89it/s][A[A[A[A[A
    
    
    
    
     80%|███████▉  | 544/682 [00:08<00:02, 58.14it/s][A[A[A[A[A
    
    
    
    
     81%|████████  | 552/682 [00:08<00:02, 62.16it/s][A[A[A[A[A
    
    
    
    
     82%|████████▏ | 560/682 [00:08<00:01, 64.26it/s][A[A[A[A[A
    
    
    
    
     83%|████████▎ | 567/682 [00:08<00:01, 65.10it/s][A[A[A[A[A
    
    
    
    
     84%|████████▍ | 574/682 [00:08<00:01, 63.80it/s][A[A[A[A[A
    
    
    
    
     85%|████████▌ | 581/682 [00:09<00:01, 64.41it/s][A[A[A[A[A
    
    
    
    
     86%|████████▌ | 588/682 [00:09<00:01, 63.15it/s][A[A[A[A[A
    
    
    
    
     87%|████████▋ | 595/682 [00:09<00:01, 62.47it/s][A[A[A[A[A
    
    
    
    
     88%|████████▊ | 602/682 [00:09<00:01, 62.21it/s][A[A[A[A[A
    
    
    
    
     89%|████████▉ | 609/682 [00:09<00:01, 63.31it/s][A[A[A[A[A
    
    
    
    
     90%|█████████ | 616/682 [00:09<00:01, 63.37it/s][A[A[A[A[A
    
    
    
    
     91%|█████████▏| 623/682 [00:09<00:00, 63.25it/s][A[A[A[A[A
    
    
    
    
     93%|█████████▎| 631/682 [00:09<00:00, 66.14it/s][A[A[A[A[A
    
    
    
    
     94%|█████████▎| 639/682 [00:09<00:00, 68.18it/s][A[A[A[A[A
    
    
    
    
     95%|█████████▍| 647/682 [00:10<00:00, 69.42it/s][A[A[A[A[A
    
    
    
    
     96%|█████████▌| 654/682 [00:10<00:00, 69.00it/s][A[A[A[A[A
    
    
    
    
     97%|█████████▋| 661/682 [00:10<00:00, 62.04it/s][A[A[A[A[A
    
    
    
    
     98%|█████████▊| 668/682 [00:10<00:00, 55.30it/s][A[A[A[A[A
    
    
    
    
     99%|█████████▉| 674/682 [00:10<00:00, 54.23it/s][A[A[A[A[A
    
    
    
    
    100%|█████████▉| 681/682 [00:10<00:00, 57.96it/s][A[A[A[A[A
    
    
    
    
    [A[A[A[A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4 
    
    CPU times: user 8 s, sys: 2.51 s, total: 10.5 s
    Wall time: 11.1 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>




## Reflections

Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!


## Submission

If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('./test_videos/challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video extra.mp4
    [MoviePy] Writing video extra.mp4


    Exception ignored in: <bound method VideoFileClip.__del__ of <moviepy.video.io.VideoFileClip.VideoFileClip object at 0x11c53efd0>>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.5/site-packages/moviepy/video/io/VideoFileClip.py", line 86, in __del__
        del self.reader
    AttributeError: reader
    Exception ignored in: <bound method VideoFileClip.__del__ of <moviepy.video.io.VideoFileClip.VideoFileClip object at 0x11da98b70>>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.5/site-packages/moviepy/video/io/VideoFileClip.py", line 86, in __del__
        del self.reader
    AttributeError: reader
    
    
    
    
    
      0%|          | 0/251 [00:00<?, ?it/s][A[A[A[A[A
    
    
    
    
      2%|▏         | 4/251 [00:00<00:06, 38.33it/s][A[A[A[A[A
    
    
    
    
      3%|▎         | 8/251 [00:00<00:06, 38.39it/s][A[A[A[A[A
    
    
    
    
      5%|▍         | 12/251 [00:00<00:06, 37.07it/s][A[A[A[A[A
    
    
    
    
      6%|▋         | 16/251 [00:00<00:06, 37.53it/s][A[A[A[A[A
    
    
    
    
      9%|▉         | 22/251 [00:00<00:05, 40.56it/s][A[A[A[A[A
    
    
    
    
     10%|█         | 26/251 [00:00<00:05, 39.49it/s][A[A[A[A[A
    
    
    
    
     12%|█▏        | 30/251 [00:00<00:05, 38.10it/s][A[A[A[A[A
    
    
    
    
     14%|█▎        | 34/251 [00:00<00:05, 36.22it/s][A[A[A[A[A
    
    
    
    
     15%|█▌        | 38/251 [00:01<00:05, 36.42it/s][A[A[A[A[A
    
    
    
    
     17%|█▋        | 43/251 [00:01<00:05, 38.55it/s][A[A[A[A[A
    
    
    
    
     19%|█▊        | 47/251 [00:01<00:06, 31.47it/s][A[A[A[A[A
    
    
    
    
     20%|██        | 51/251 [00:01<00:06, 31.18it/s][A[A[A[A[A
    
    
    
    
     22%|██▏       | 55/251 [00:01<00:07, 25.62it/s][A[A[A[A[A
    
    
    
    
     23%|██▎       | 58/251 [00:01<00:07, 24.91it/s][A[A[A[A[A
    
    
    
    
     24%|██▍       | 61/251 [00:01<00:07, 25.40it/s][A[A[A[A[A
    
    
    
    
     25%|██▌       | 64/251 [00:01<00:07, 26.22it/s][A[A[A[A[A
    
    
    
    
     27%|██▋       | 67/251 [00:02<00:06, 26.57it/s][A[A[A[A[A
    
    
    
    
     28%|██▊       | 70/251 [00:02<00:06, 25.90it/s][A[A[A[A[A
    
    
    
    
     29%|██▉       | 73/251 [00:02<00:06, 25.84it/s][A[A[A[A[A
    
    
    
    
     30%|███       | 76/251 [00:02<00:06, 26.00it/s][A[A[A[A[A
    
    
    
    
     32%|███▏      | 80/251 [00:02<00:06, 27.26it/s][A[A[A[A[A
    
    
    
    
     33%|███▎      | 83/251 [00:02<00:06, 24.19it/s][A[A[A[A[A
    
    
    
    
     34%|███▍      | 86/251 [00:02<00:06, 23.60it/s][A[A[A[A[A
    
    
    
    
     35%|███▌      | 89/251 [00:02<00:06, 23.94it/s][A[A[A[A[A
    
    
    
    
     37%|███▋      | 92/251 [00:03<00:06, 23.34it/s][A[A[A[A[A
    
    
    
    
     38%|███▊      | 95/251 [00:03<00:07, 22.03it/s][A[A[A[A[A
    
    
    
    
     39%|███▉      | 98/251 [00:03<00:07, 21.72it/s][A[A[A[A[A
    
    
    
    
     40%|████      | 101/251 [00:03<00:06, 22.44it/s][A[A[A[A[A
    
    
    
    
     41%|████▏     | 104/251 [00:03<00:06, 22.01it/s][A[A[A[A[A
    
    
    
    
     43%|████▎     | 107/251 [00:03<00:06, 21.15it/s][A[A[A[A[A
    
    
    
    
     44%|████▍     | 110/251 [00:03<00:06, 21.61it/s][A[A[A[A[A
    
    
    
    
     45%|████▌     | 113/251 [00:04<00:06, 21.98it/s][A[A[A[A[A
    
    
    
    
     47%|████▋     | 117/251 [00:04<00:05, 24.27it/s][A[A[A[A[A
    
    
    
    
     48%|████▊     | 120/251 [00:04<00:05, 24.57it/s][A[A[A[A[A
    
    
    
    
     49%|████▉     | 123/251 [00:04<00:05, 23.30it/s][A[A[A[A[A
    
    
    
    
     50%|█████     | 126/251 [00:04<00:05, 22.12it/s][A[A[A[A[A
    
    
    
    
     51%|█████▏    | 129/251 [00:04<00:05, 21.48it/s][A[A[A[A[A
    
    
    
    
     53%|█████▎    | 132/251 [00:04<00:05, 20.62it/s][A[A[A[A[A
    
    
    
    
     54%|█████▍    | 135/251 [00:05<00:05, 21.40it/s][A[A[A[A[A
    
    
    
    
     55%|█████▍    | 138/251 [00:05<00:05, 21.50it/s][A[A[A[A[A
    
    
    
    
     56%|█████▌    | 141/251 [00:05<00:05, 19.86it/s][A[A[A[A[A
    
    
    
    
     57%|█████▋    | 144/251 [00:05<00:05, 20.09it/s][A[A[A[A[A
    
    
    
    
     59%|█████▊    | 147/251 [00:05<00:05, 19.56it/s][A[A[A[A[A
    
    
    
    
     60%|█████▉    | 150/251 [00:05<00:04, 20.33it/s][A[A[A[A[A
    
    
    
    
     61%|██████    | 153/251 [00:05<00:04, 21.23it/s][A[A[A[A[A
    
    
    
    
     62%|██████▏   | 156/251 [00:06<00:04, 20.26it/s][A[A[A[A[A
    
    
    
    
     63%|██████▎   | 159/251 [00:06<00:04, 20.82it/s][A[A[A[A[A
    
    
    
    
     65%|██████▍   | 162/251 [00:06<00:04, 22.12it/s][A[A[A[A[A
    
    
    
    
     66%|██████▌   | 165/251 [00:06<00:03, 23.50it/s][A[A[A[A[A
    
    
    
    
     67%|██████▋   | 169/251 [00:06<00:03, 23.96it/s][A[A[A[A[A
    
    
    
    
     69%|██████▊   | 172/251 [00:06<00:03, 22.16it/s][A[A[A[A[A
    
    
    
    
     70%|██████▉   | 175/251 [00:06<00:03, 22.06it/s][A[A[A[A[A
    
    
    
    
     71%|███████   | 178/251 [00:07<00:03, 22.66it/s][A[A[A[A[A
    
    
    
    
     72%|███████▏  | 181/251 [00:07<00:03, 22.25it/s][A[A[A[A[A
    
    
    
    
     73%|███████▎  | 184/251 [00:07<00:02, 22.87it/s][A[A[A[A[A
    
    
    
    
     75%|███████▍  | 187/251 [00:07<00:02, 23.90it/s][A[A[A[A[A
    
    
    
    
     76%|███████▌  | 190/251 [00:07<00:02, 24.01it/s][A[A[A[A[A
    
    
    
    
     77%|███████▋  | 193/251 [00:07<00:02, 23.61it/s][A[A[A[A[A
    
    
    
    
     78%|███████▊  | 196/251 [00:07<00:02, 23.27it/s][A[A[A[A[A
    
    
    
    
     80%|███████▉  | 200/251 [00:07<00:02, 25.17it/s][A[A[A[A[A
    
    
    
    
     81%|████████  | 203/251 [00:08<00:01, 24.87it/s][A[A[A[A[A
    
    
    
    
     82%|████████▏ | 206/251 [00:08<00:01, 26.11it/s][A[A[A[A[A
    
    
    
    
     83%|████████▎ | 209/251 [00:08<00:01, 23.98it/s][A[A[A[A[A
    
    
    
    
     84%|████████▍ | 212/251 [00:08<00:01, 24.99it/s][A[A[A[A[A
    
    
    
    
     86%|████████▌ | 216/251 [00:08<00:01, 27.00it/s][A[A[A[A[A
    
    
    
    
     88%|████████▊ | 220/251 [00:08<00:01, 28.87it/s][A[A[A[A[A
    
    
    
    
     89%|████████▉ | 223/251 [00:08<00:00, 29.09it/s][A[A[A[A[A
    
    
    
    
     90%|█████████ | 227/251 [00:08<00:00, 30.63it/s][A[A[A[A[A
    
    
    
    
     92%|█████████▏| 232/251 [00:09<00:00, 32.59it/s][A[A[A[A[A
    
    
    
    
     94%|█████████▍| 236/251 [00:09<00:00, 32.21it/s][A[A[A[A[A
    
    
    
    
     96%|█████████▌| 240/251 [00:09<00:00, 32.34it/s][A[A[A[A[A
    
    
    
    
     97%|█████████▋| 244/251 [00:09<00:00, 30.94it/s][A[A[A[A[A
    
    
    
    
     99%|█████████▉| 248/251 [00:09<00:00, 31.86it/s][A[A[A[A[A
    
    
    
    
    100%|██████████| 251/251 [00:09<00:00, 26.03it/s][A[A[A[A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: extra.mp4 
    
    CPU times: user 6.7 s, sys: 1.53 s, total: 8.23 s
    Wall time: 10.5 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="extra.mp4">
</video>




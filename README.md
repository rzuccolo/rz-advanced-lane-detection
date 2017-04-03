# Finding Lane Lines on the Road - Advanced Techniques

## Computer vision with OpenCV

### Join me on this exciting journey to apply advanced computer vision techniques to identify lane lines. Camera calibration,  undistortion, color threshold, perspective transformation, lane detection and image annotation. Thanks to Udacity Self-driving Car Nanodegree for providing me the basic skills set to get there!


#### When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are, will act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

#### In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.

---

**Advanced Lane Finding Project**

The goals/ steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Images References)

[image1]: ./my_images/15-image-pipeline-readme.png "Finding Lane Line"



![alt text][image1] 


---

## Code Files & Functionality

### 1. Files:

* **camera_calibration.py**  is the script used to analyze the set of chessboard images, and save the camera calibration coefficients (mtx,dist). That is the first step in the project.
* **perspective_transform.py** is the script used to choose the appropriate perspective transformation matrices (M, Minv) to convert images to bird's-eye view. That is the second step in the project.
* **load_parameters.py** contains the functions used to load camera coefficients (mtx, dist) and perspective matrices (M, Minv).
* **warp_transformer.py** contains the functions used to color transform, warp and create the binary images.
* **line.py** defines a class to keep track of lane line detection. This information helps the main pipeline code to decide if the current image frame is good or bad.
* **main.py** contains the script used to run the video pipeline and create the final annotated video.
* **[Annotated Project Video](https://vimeo.com/211246515)** Click on this link to watch the annotations for the project video.
* **[Annotated Challenge Video](https://vimeo.com/211246891)** Click on this link to watch the annotations for the challenge video!
* **writeup_report.md** is the summary report of the results



### 2. Functional codes:

#### Camera calibration:
Open the **camera_calibration.py** and set the proper output directory (code lines 8-17).

Default configuration will:
* Read calibration images from: `camera_cal`
* Save results to: `output_images/camera_cal` 

Execute the script as follow: 
```
python camera_calibration.py
```

#### Perspective Transform Matrices:
Open the **load_parameters.py** and set the proper income directories (code lines 5-11).

Default configuration will:
* Read camera coefficients from: `output_images/camera_cal`

Open the **perspective_transform.py** and set the proper output directory (code lines 12-21).

Default configuration will:
* Save warped straight road image for check to: `output_images/bird_eye_test`
* Save results perspective matrices to: `output_images/camera_cal`

Execute the script as follow: 
```
python perspective_transform.py
```
Modify the trapezoid ratios (code lines 28-31) until you are happy with the output bird's-eye image 


#### Video Pipeline:
Open the **load_parameters.py** and set the proper income directories (code lines 5-11).

Default configuration will:
* Read camera coefficients and perspective matrices from: `output_images/camera_cal`

Open the **main.py** and set the proper output directory and video source (code lines 303-304).

Default configuration will:
* Read video source from parent directory
* Save annotated video to: `output_images`

Execute the script as follow: 
```
python main.py
```


---


## Discussion

I found the thresholding technique very challenge to generalize for non-well maintained or under construction roads, and for tracks with very sharp curves. So I guess there are more advanced techniques out there, restraining and smoothing big detection variations. Regarding the sharp curves, I guess in this case we are limited by the field of view of just one camera, but it may still be doable if we use the appropriate transformation matrix.

As described above:
* I had no time to play with windows margins, but it would be interesting to come back later and try out some modifications. It also may be interesting to insert one more variable of control and limit the horizontal shift of the "centroid" with respect to the last window, it may avoid crazy windows combination and ultimately wrong lane detection.
* I had no time as well to implement a lose track condition. Let's say it cannot find non-zero pixels within the windows, and it may happen for a sequence of frames, the program should go back to the sliding windows search or another method to rediscover them.

The pipeline is doing great on the project video!! it is doing good on the challenge videos! But it fails badly on the brutal "harder_challenge" video. In my opinion, first, because it switches from a wide one way (several lanes) road to a much narrower (one lane) and two ways road. Second, because of the very sharp curves going out of the field of view of the camera.

I did not stop yet to try and improve the pipeline, generalizing up to the harder challenge video. But I think the key is to implement a dynamic and automated way to define the transformation trapezoid and come up with the appropriate source points for the perspective transform. I would start looking into the edge or corner detection options and analyze attributes like color and surrounding pixels. 


---


## Acknowledgments / References

* [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive)
* [David A. Ventimiglia](http://davidaventimiglia.com/advanced_lane_lines.html)
* [Vivek Yadav](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa)

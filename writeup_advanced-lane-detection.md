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

[//]: # (Image References)

[image1]: ./my_images/1-camera_cal.png "Undistorted Chessboard"
[image2]: ./my_images/2-camera_cal_road_original.png "Road Images - Original"
[image3]: ./my_images/3-camera_cal_road_undistorted.png "Road Images - Undistorted"
[image4]: ./my_images/4-bird_eye.png "Birds-eye View"
[image5]: ./my_images/5-src_dst_points.png "Transformation - Source (src) and Destination (dst) points"
[image6]: ./my_images/6-threshold_sx_hsl.png "Gradient & Color Thresholding - Sx and HSL Saturation"
[image7]: ./my_images/7-threshold_yellow-white.png "Color Thresholding - Yellow-White"
[image8]: ./my_images/8-threshold_sx_yellow-white.png "Gradient & Color Thresholding - Sx and Yellow-White"
[image9]: ./my_images/9-histogram.png "Peak Detection by Histogram"
[image10]: ./my_images/10-sliding_window.png "Sliding Window Method"
[image11]: ./my_images/11-no_sliding_window.png "Window Search"
[image12]: ./my_images/12-curvature_theory.png "Radius of Curvature Theory"
[image13]: ./my_images/13-radius-offset.png "Radius and Offset Annotations"
[image14]: ./my_images/14-images-pipeline.png "Pipeline on Images"


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


Playground using Jupyter notebooks for all stages of the projects can be found here:
**[github repository](https://github.com/rzuccolo/rz-advanced-lane-detection)**



### 2. Functional codes:

#### 2.1 Camera calibration:
Open the **camera_calibration.py** and set the proper output directory (code lines 8-17).

Default configuration will:
* Read calibration images from: `camera_cal`
* Save results to: `output_images/camera_cal` 

Execute the script as follow: 
```
python camera_calibration.py
```

#### 2.2 Perspective Transform Matrices:
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


#### 2.3 Video Pipeline:
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


## Camera Calibration

### 1. How camera matrix and distortion coefficients are computed:

The code for this step is contained in **camera_calibration.py**.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Chessboard:

![alt text][image1]


Road test images - Original:

![alt text][image2]


Road test images - Undistorted:

![alt text][image3]


---


## Bird's-Eye View Transformation

### 1. How bird's-eye view is optimized and computed:

The code for this step is contained in **perspective_transform.py**.  

We want to measure the curvature of the lines and to do that, we need to transform the road image to a top-down view. To Compute the perspective transform, M, given the source and destination points we use `cv2.getPerspectiveTransform(src, dst)`. To compute the inverse perspective transform we use `cv2.getPerspectiveTransform(dst, src)`. Finally, we can Warp the image using the perspective transform `cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`.

We need to identify four source points for the perspective transform. In this case, I assumed that road is a flat plane. This isn't strictly true, but it can serve as an approximation for this project. We need to pick four points in a trapezoidal shape (similar to region masking) that would represent a rectangle when looking down on the road from above.

There are many ways to select it. For example, many perspective transform algorithms will programmatically detect four source points in an image based on the edge or corner detection, and analyze attributes like color and surrounding pixels. I have selected a trapezoid by using image dimensions ratios as an input. I found it to be a smart way to manually calibrate the pipeline and make sure it generalizes for different roads. It is the same trapezoid function used in the first (simplified) lane detection project.
(code lines 74-95)

I have also implemented a code to properly sort the four source points for the perspective transformation. Just in case we change the way we come up with those points in the future. It is VERY IMPORTANT to feed it correctly, a wrong step here will mess everything up. The points need to be sorted "clock-wise", starting from top-left. The methodology consists in normalize the input into the [0, 2pi] space, which naturally will sort it "counter-clockwise". Then I inverse the order in the function return. (code lines 34-48)

How we make sure we have a good transformation?
The easiest way to do this is to investigate an image where the lane lines are straight and find four points lying along the lines that, after perspective transform, make the lines look straight and vertical from a bird's-eye view perspective.
I applied undistortion and then bird's-eye transformation on a straight image of the road, and played with the trapezoid dimensions until getting this result: 

Final trapezoid ratios and car's hood cropping:

* **bottom_width=0.4**, percentage of image width
* **top_width=0.092**, percentage of image width
* **height=0.4**, percentage of image height
* **car_hood=45**, number of pixels to be cropped from bottom meant to get rid of car's hood

![alt text][image4]


Here are the source (src) and destination (dst) points:

![alt text][image5]


---


## Image Thresholding, Binary Image

### 1. How the warped binary image is computed:

The code for this step is contained in **warp_transformer.py**.  

If you check my playground notebook for thresholding you will see I have tried out various combinations of color and gradient thresholds to generate a binary image where the lane lines are clearly visible. There's more than one way to achieve a good result, but I have achieved the best results using only color threshold. Running my pipeline on the challenge video made clear that using color threshold was the best in my case.   

While using gradient threshold I found Sobel gradient in X (Sx) and gradient magnitude (square root of the squares of the individual x and y gradients) were the best approaches to make the lines clear. When Sx operator is applied to a region of the image, it identifies values that are rising from left to right, so taking the gradient in the x-direction emphasizes edges closer to vertical. 

`sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)`

`sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)`

`abs_sobelx = np.absolute(sobelx)`

Convert the absolute value image to 8-bit:

`scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))`

The color thresholding was the best overall for me. I started with the recommended HSL thresholding method but ended up with an approach I have used in the first detection project. My approach is to mask the image with white and yellow colors, convert it to grayscale and create a binary image based on non-zero pixels. The trade-off for better quality detection was the insertion of color constraints into the lane system detection. (code lines 74-95)

Yellow-White color threshold that best generalized was:
* **Yellow: HSV [50,50,50] to HSV [110,255,255]**
* **White: RGB [200,200,200] to RGB [255,255,255]**



Bird's-eye view for Sobel absolute gradient X(scaled_sobel 15 to 100) and HSL(S channel 170 to 255) thresholding:

![alt text][image6]


Bird's-eye view for Yellow(HSV[90,100,100] to HSV[110,255,255]) and White(RGB200 to RGB255) thresholding:

![alt text][image7]


Bird's-eye view for Sobel absolute gradient X(scaled_sobel 15 to 100) Yellow(HSV[90,100,100] to HSV[110,255,255]) and White(RGB200 to RGB255) thresholding:

![alt text][image8]


---


## Main Pipeline Video

### 1. Identify Lane Lines

The code for this step is contained in **main.py**.  

Next, locate the Lane Lines and fit a polynomial. Here are the steps:
* Identify start bottom image position of right and left lines: **Peaks in a Histogram method**.
* Search for the biggest "accumulation of 1s" in horizontal "slices" of the binary image and define a window. **Sliding Window method**.
* Identify the nonzero pixels in x and y within the window and fit a second order polynomial to each line. **Polyfit**.

#### 1.1 Peaks in a Histogram Method

After applying calibration, thresholding, and a perspective transform to a road image, we have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image. (Code line 17)

![alt text][image9]


With this histogram, I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I use that as a starting point for where to search for the lines.


#### 1.2 Sliding Window Method

Next, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
We basically search for the biggest "accumulation of 1s" in horizontal "slices" of the binary image and define a window.

9 windows are stacked up along the lane line (each), i.e. the binary image is "sliced and searched" in 9 blocks from bottom to top of the image, but non-zero points will be searched only with the delimited windows (right and left). The initial "centroid" of the windows for the first block is defined by the histogram method. 100 pixels width margin is used to search non-zero pixels within the window. I case the "accumulation" of non-zero points is bigger than 50, the "centroid" of the window is redefined accordingly.

I had no time to play with windows margin (100, 50), but it will be interesting to come back later and try out some modifications. It also may be interesting insert one more variable of control and limit the horizontal shift of the "centroid" with respect to the last window, it may avoid crazy windows combination and ultimately wrong lane detection.

Polyfit and drawing are applied next.

(code lines 15-98)

![alt text][image10]



#### 1.3 Non-Sliding Window (Window Search)

The sliding window method is applied to the first frame only. After that we expect the next frame to have a very similar shape, so we can simply search within the last windows and adjust the "centroids" as necessary.

Polyfit and drawing are applied next.

(code lines 121-158)

![alt text][image11]


The green shaded area shows where it searches for the lines. So, once you know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame. This is equivalent to using a customized region of interest for each frame of video, which helps to track the lanes through sharp curves and tricky conditions.

I had no time to implement a lose track condition yet. Let's say it cannot find non-zero pixels within the last windows, and it may happen for a sequence of frames, the program should go back to the sliding windows search or another method to rediscover them.



#### 1.4 Good and bad polyfit frames

I have implemented 2 ways to avoid the crash of the program and discard bad polyfits. First I discard the frame if the polyfit crashes. Second, I keep track of the last line polyfit and calculate the difference to the current. We expect the current frame to have a very similar line shape to the last one. So I have inserted polyfit coefficients tolerances.

How did I come up with the tolerances?
As an initial point I have fitted lines for several frames in different scenarios, and calculate the min and max differences between those different scenes, That gives us initial numbers but the margin is still large because the frames should be very similar and hence smaller tolerances. But I thought about cases in which it will be discarding a relatively long sequence of frames, so the next not discarded frame may be within that range of tolerances. 

We have 2nd order polyfit, so we have 3 coefficients. **The tolerances used are: 0.001, 0.4, 150.**
(code lines 138-158)




### 2. Radius of Curvature and Offset Position

#### 2.1 Radius of curvature

(code lines 100-115) for sliding window
(code lines 160-180) for nonsliding window

Next, we'll compute the radius of curvature of the fit and the car offset position with respect to lane center.
The radius of curvature is defined as follow:

[Radius of Curvature - M. Bourne](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

![alt text][image12]

The y values of the image increase from top to bottom, I chose to measure the radius of curvature closest to your vehicle, so we evaluate the formula above at the y value corresponding to the bottom of the image, or in Python, at yvalue = image.shape[0].

If we calculate the radius of curvature based on pixel values, the radius will be in pixel space, which is not the same as real world space. So we first convert x and y values to real world space. 

The conversion to real world could involve the measuring how long and wide the section of the lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane's dimensions in the field of view of the camera, but for this project, we assume the lane is about 30 meters long and 3.7 meters wide.

For future developments, we could derive a conversion from pixel space to world space using images, compare those images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each. [U.S. government specifications for highway curvature](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC)

Here is an example of my result on a test image:

![alt text][image13]


#### 2.2 Offset position

(code lines 202-204)

For this project, we assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines we've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.



### 3. Average Frame (Smooth)

(code lines 234-267)

Even when everything is working, the line detections will jump around from frame to frame a bit and it is preferable to smooth over the last n frames of video to obtain a cleaner result. Each time we get a new high-confidence measurement, we append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position we want to draw onto the image.

The good or bad frame selection is already implemented as described above. For the frame buffer and average I found a really helpful implementation by [David A. Ventimiglia](http://davidaventimiglia.com/advanced_lane_lines.html):


>*"Using a ring-buffer with the Python deque data structure along with the Numpy average function made it very easy to implement a weighted average over some number of previous frames. Not only did this smooth out the line detections, lane drawings, and distance calculations, it also had the added benefit of significantly increasing the robustness of the whole pipeline. Without buffering—and without a mechanism for identifying and discarding bad detections—the lane would often bend and swirl in odd directions as it became confused by spurious data from shadows, road discolorations, etc. With buffering almost all of that went away, even without discarding bad detections..."*


### 4. Road Test Images

![alt text][image14]



### 5. Road Test Videos!

* **[Annotated Project Video](https://vimeo.com/211246515)** Click on this link to watch the annotations for the project video.
* **[Annotated Challenge Video](https://vimeo.com/211246891)** Click on this link to watch the annotations for the challenge


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

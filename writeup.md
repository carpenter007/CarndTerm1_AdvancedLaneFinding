# Advanced Lane Finding Project

## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./output_images/camera_calibration.jpg "Undistorted Chessfield"
[image2]: ./output_images/undistorted_test_image.jpg "Undistorted test image"

[video1]: ./project_video.mp4 "Video"


---

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./lanefinding.ipynb".  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The camera matrix and the distortion coefficients are stored in "cameraCorrection.p" with pickle.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The camera correction parameters are stored as pickle data in 'cameraCorrection.p'

---
[//]: # (Image References)
[image2]: ./output_images/undistorted_test_image.jpg "Undistorted test image"
[image3]: ./output_images/test_image_binarys.jpg "different binary transformations"
[image4]: ./output_images/test_image_transformed.jpg "image transformed to bird-eye view"
[image5]: ./output_images/lane_extraction.jpg "extracted lanes from binary image"
[image6]: ./output_images/identified_lane.jpg "identified lane plotted on undistorted image"
[image7]: ./output_images/radius_and_center_offset.jpg "radius and center offset"

### Pipeline (single images) 

#### Image distortion-correction
By using the stored camera correction coefficients, the images will be undistorted.

![alt text][image2]


#### Perform a perspective transform

An importent step in the image pipeline is the perspective transform from front view to bird view. This is done by defining a isosceles trapezoid in the image area where a straight road is expected to be found. Transforming this image into a rectangle gives me a birdview of the road. Only in the birdview, the lane lines are printed parallel to each other. In this perspective I will be able to calculate the radius of a curve.

![alt text][image4]


#### Create thresholded binary image

There are different methods to extract the edges, which are potentially lane lines, and display them in a binary image. Different lane line colors, shapes and lengths need to be recognizeds. The images also can include shaddows or reflections and so on. Therefore I create more than one binary image and combine them in the end. I implemented a lot of binary files with different thresholds. The all have different weaknesses and strengths.
These are the most promising thresholds:
- Using the sobel filter with a gradient threshold of 32 to 95 in x orientation ('''gradx''')
- Using the sobel filter with a gradient threshold of 20 to 100 in y orientation ('''grady''')
- Creating a magnitued binary which represents the absolut values of the sobel filter output (magnitude of gradient) in x and y direction ('''mag_binary_bv''')
- Using a direction filtered binary image, which only displays the direction of the magnitued which has a slope of 0.7 to 1.3 ('''dir_binary_bv''')
- Using the S-Channel in a HLS color space converted image helps me to extract colored (yellow) lines ('''sHls_binary''')
- Creating a binary image of the S-channel of a HSV converted image with threshold 190 to 220 ('''sHsv_binary''')
- Extratcting the L-channel of LUV with a threshold from 200 to 255 ('''lLuv_binary''')
- Using B-channel of LAB converted image with a threshold from 145 to 200 ('''bLab_binary''')

All these image transformations can be found in several defined functions in code cell 2 of the lanefinding.ipynb notebook.

I extracted more frames out of the given videos and compared the different binary images to each other. While LAB, HLS, HSV and LUV converted images are great to detect the white and lanes based on colors, the gradient thresholded binaries are great to detect edges. The magnitued binary and the direction filtered binary images help to filter out false findings.

At the end I used following combination to get a resulting binary image:

```python
combined[(((sHls_binary == 1) | (lLuv_binary == 1)) \
                & ((dir_binary_bv == 1) | (mag_binary_bv == 1) | (gradx == 1))) | (bLab_binary == 1)] = 1
```

After some tests I decided, first to do the perspective transformation and afterwards creating the binaries. Especially for the color based binaries the result is smarter, with less noise at the top of the image. 
While the color based binaries find lines even with rough street and light conditions, there are sometimes a huge number of pixels which are marked in the binary image, but which are obviously not part of the lane. The the gradient in x direction (of the perspective transformed image), the magnitue and the direction of the magnitude help to verify the pixels on the binary image.

![alt text][image3]


#### Identify lane-line pixels

This is done with two different functions. Initially I use a more expensive function to identify the lane-line pixels. I visualize the peak of the left and right halves of the histogram in the bottom half of the image. These will be the starting point for the left and right lines. By sliding a fixed size window over the perspective-transformed binary image, I can detect accumulations of pixels in the image. The pixels within nine oriented fix sized windows are used to extract one polynom (ax² + bx + c) for each line.

![alt text][image5]

When we are focusing on self-driving cars, we don't want to edit videos but we want to process images at real time. Therefore it is important to use performance optimized techniques when extracting the lane lines. Because of this, once we found a the lane-lines on a image, we expect the lane-lines of the next frame of the video to be at least in the area around the previous lane-lines. Using this information helps to focus only on a small part of the image when trying to detect the lane-lines on further frames.

Calling the sliding_window_line_extraction with polynom coefficients for the left and right lane will lead to the performance optimized line extraction. 

```python
def sliding_window_line_extraction(img, nwindows = 9, minpix = 50, \
                                   margin = 100, left_fit= [0,0,0], \
                                   right_fit = [0,0,0]):
```
#### Calculate the radius of curvatur and car position

Next step is to calculate useful numbers out of the extracted lane lines. Two useful numbers are the radius of the curvatur and the car position relative to the center of the current lane.

First we need to convert numbers of pixels into meters or inches. According to the lecutres of Udacity, I can assume that the street has lane width of 3.7meters (based on U.S. regulations), and the lane is detected about a length of 30 meters.

https://www.intmath.com/applications-differentiation/8-radius-curvature.php visualizes how to calculate the radius of a curve on any point.

The code to calculate the cars position can be found in the function 'measure_radius_of_curvature' in the second code cell of the lanefinding.ipynb notebook.

The cars position relative to the center of the current lane can be calculated by calculating the offset of both lanes to the middle of the image.
 
```python
# compute the offset from the center
lane_center = (right_fitx[-1] + left_fitx[-1])/2
center_offset_pixels = abs(undistimg.shape[1]/2 - lane_center)
center_offset_mtrs = xm_per_pix*center_offset_pixels
```

![alt text][image7]

#### Plot the identified lane

At the end of the pipeline, the found polynom for the lane is drawn onto a warped blank image. Then I warp the blank back to original image space using inverse perspective matrix ('Minv')

The following picture shows the result which is a combination of the original image and the new wraped image with the found lane.

```python
result = cv2.addWeighted(undistimg, 1, newwarp, 0.3, 0)
```

![alt text][image6]

---

[image8]: ./output_images/reasons_for_invalidating_frames.jpg "Printed reasons for invalidating frames"
[video2]: ./output_video/challenge_video.mp4 "output for challenge video"

### Pipeline (video) 

I've tested the pipeline with two of the given videos ('project_video.mp4' and 'challenge_video.mp4'). They both use the same pipeline. While the 'project_video' output was quite good only by optimizing the binary images, the 'challenge_video' forced me to smooth and validate the output with the help of previous frames of the video.

Using a class called 'Line' (can be found in the 3rd code cell of the lanefinding.ipynb notebook), I record information about current and previous pixels and polynom coefficients of detected lanes. There is one object of the Line class for each lane 'Left' and 'Right'.

To get a smoother output video, I use the mean of 10 lanes for calculating the resulting polynom.

To verify each lane by it self, I check the latestly found lane against the previous calculated lane. If the coefficients differ to much, the frame will be ignored.

Further a sanity check is done by comparing both lines (Left and Right):
- Checking that they have similar curvature and that they are roughly parallel
- Checking that they are separated by approximately the right distance horizontally

The following picture shows a debug print while creating a output video. There is the reason printed for each frame which hasn't been validated. Each frame number which has an extra output is sorted out.
![alt text][image8]

The following video shows the output of the challenge video.
![alt text][video2]

---

### Discussion

The pipeline fits good to two of the given videos, but it still admits a lot of room of improvement. 
The harder_challange_video.mp4 (which can be found in the Gitrepository as well) introduces new challenging frames.

- Second degree polynominals are not fitting any longer if the road bends sharply in left and right direction consecutively.
- Colored based thresholds are good on defined conditions, but the can easily fail on other conditions since they are defined statically.
- The sanity check can be improved with less effort. Processing specific failing images and printing their detected coefficients (together with the previous good coefficients) will improve the result even for harder road videos
- perspective transforming could be done more flexible
- If one lane is detected very reliable, then the other lane could be calculated by given knowledge of the road (parallel lanes, width of the lane)
- The performance could be improved at several places (e.g. image is converted to grayscale several times in the pipeline because there are several functions for each transformation)
- I tried out a lot of different colorspaces and combined their binary outputs with the magnitude / direction / graduation binary outputs. But there are still a lot of ways to explore. E.g. using gradient / magnetude / direction thresholds directly on several color converted images and channels, instead of using these filters only on grayscaled channels.
- Using machine learning techniques to detect the lane could be implemented


Thanks to Udacity for this nice project. Thanks to the Slack community for helpful disussions.

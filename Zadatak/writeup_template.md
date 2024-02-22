**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Info

Prroject consists out of few files:
    cameraCalibration.py
    imageProcessingMethods.py
    laneDetection.py
    imageProcessing.py
    videoProcessing.py
    main.py

Throughout the documentation it will be explained the purpose of each file

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code located in "scripts/cameraCalibration.py".  

The process start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here is assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time it successfully detects all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then the output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. It applies this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

[image1]: .\output\cameraCalibration\examplePair.png

### Pipeline (single images)

#### 1. Undistortion

In single images pipeline we start by using basic methods that are implemented in the python script “./imageProcessingMethods.py”. Undistortion is implemented there and it is called for every input test image there is. This function takes arguments of an input image, ‘mtx’ and ‘dist’ results of calibration, and ‘src’ and ‘dst’ points.

#### 2. Start of preparing image for processing

Getting to thresholded binary image took a few steps. Image processing is done in file imageProcessing.py which uses imageProcessingMethods.py to use functios for HSV, threshold and more. After the undistortion we process the image to HSV where our target is to leave yellow and white objects. After applying the Gaussian and warping the image to different perspective(we talk about this later in document), threshold is used to get the binary image prepared for later processing.

[undistorted HSV image]: .\output\imageProcessing\challange00101.jpg_processed_1.jpg

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warpImage()` which is also in imageProcessingMethods.py, and is used in imageProcessing.py after the use of gaussian.  The `warpImage()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner based on the image resolution.

```python
 src = np.float32(
                [[480, 342],
                 [260, 540],
                 [800, 540],
                 [557, 342]
                 ])
            dst = np.float32(
                [[264, 0],
                 [264, 540],
                 [831, 540],
                 [831, 0]
                 ])

        elif img.shape == (720, 1280, 3):
            print("drugi if")
            src = np.float32(
                [[587, 464],
                 [383, 695],
                 [1030, 695],
                 [718, 464],
                 ])
            dst = np.float32(
                [[320, 0],
                 [320, 720],
                 [960, 720],
                 [960, 0]
                 ])
```

It is verified that my perspective transform is ok by drawing the `src` and `dst` points onto warped counterpart to verify that the lines appear parallel in the warped image.

[warped HSV image with verification lines]: .\output\imageProcessing\challange00101.jpg_processed_3.jpg 
[threshold image]: .\output\imageProcessing\challange00101.jpg_processed_2.jpg

#### 4. Identifing lane-line pixels and fitting their positions with a polynomial

The fit_polynomial(binary_warped) is designed for detecting and fitting polynomials to the left and right lane lines in a binary image obtained through thresholding and perspective transformation. Implementation is in laneDetection.py. It utilizes a sliding window approach, dividing the image vertically into sections and iteratively searching for lane pixels within each window. By computing histograms and identifying peaks, it initializes the starting points for the left and right lanes. Through a process of window repositioning based on detected lane pixels, the function collects indices of lane pixels and fits second-order polynomials to these points. Finally, it returns the coefficients of the fitted polynomials, providing essential information for further lane line analysis and visualization.

#### 5. Calculation of radius of curvature of the lane and the position of the vehicle with respect to center.

Function measure_curvature_real(binary_warped, left_fit, right_fit) computes the real-world curvature of the lane lines detected in a binary warped image. Implementation is in laneDetection.py. It takes the fitted polynomials of the left and right lane lines (left_fit and right_fit) along with the binary warped image (binary_warped) as inputs. By converting pixel coordinates to meters, it fits new polynomials to the lane lines in real-world space and calculates their curvature using the radius of curvature formula.
Function vehicle_position(left_fit, right_fit, binary_warped) determines the position of the vehicle relative to the center of the lane. Implementation is in laneDetection.py. It takes the fitted polynomials of the left and right lane lines (left_fit and right_fit) along with the binary warped image (binary_warped) as inputs. By calculating the x-intercepts of the lane lines at the bottom of the image and finding their midpoint, it computes the position of the vehicle. Finally, it converts the distance from the center of the lane from pixels to meters, providing a measure of the vehicle's lateral displacement.

#### 6. Image examples for lane detection, curvature of lane and the position of vehicle

[Image with curvature,vehicle position and detected lane]: .\output\imageProcessing\challange00101.jpg_processed_4.jpg

### Pipeline (video)

#### 1. Video processing

Algorithm is the same as image processing.

Here's a [link to my video result](./project_video.mp4)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

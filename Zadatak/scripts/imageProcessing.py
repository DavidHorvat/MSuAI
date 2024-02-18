import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

def warpImage(image,src_arg,dst_arg):

    imgSize = (image.shape[1], image.shape[0])
    M = cv.getPerspectiveTransform(src_arg, dst_arg)
    warpedImage = cv.warpPerspective(image, M, imgSize, flags = cv.INTER_NEAREST)  # keep same size as input image

    return warpedImage

def undistortImage(image, mtx, dist):
    h, w = image.shape[:2]

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    # undistort
    undistortedImage = cv.undistort(image, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    undistortedImage = undistortedImage[y:y+h, x:x+w]
    
    return undistortedImage

def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def gaussian(image):
    return cv.GaussianBlur(image, (5, 5), 0)

def threshold(image, thresh1, thresh2):
    # return cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv.THRESH_BINARY,9,3)
    return cv.threshold(image, thresh1, thresh2, cv.THRESH_BINARY)

def cannyTransformation(image, canny_t1, canny_t2):
    return cv.Canny(image, canny_t1, canny_t2)

def houghLines(image, rho, theta, hough_t, min_line_length, max_line_gap):
    return cv.HoughLinesP(image, rho, theta, threshold=hough_t, minLineLength=min_line_length, maxLineGap=max_line_gap)

def hsv(image):
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Mask for white color
    lower = np.array([0, 0, 200])
    upper = np.array([255, 30, 255])
    maskW = cv.inRange(hsvImage, lower, upper)

    # Mask for yellow color
    lower = np.array([10, 100, 100])
    upper = np.array([30, 255, 255])
    maskY = cv.inRange(hsvImage, lower, upper)

    mask = cv.bitwise_xor(maskW, maskY)

    filteredImage = cv.bitwise_and(image, image, mask = mask)

    return filteredImage

def drawLines(image, lines):
    # Create a copy of the original image
    image_with_lines = np.copy(image)
    
    # Iterate over the detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line endpoints
        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the line
        
    return image_with_lines

def fit_polynomial(binary_warped):
    # Assuming 'binary_warped' is the binary image after thresholding and perspective transform

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Define the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
        return None, None  # Return None if no lane pixels found
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit


def measure_curvature_real(binary_warped,left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2], 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2], 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def vehicle_position(left_fit, right_fit, binary_warped):
    # Calculate vehicle center
    center = binary_warped.shape[1] / 2
    
    # Define conversions in x from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Extract the x-intercepts of the lines
    left_x_intercept = left_fit[0]*binary_warped.shape[0]**2 + left_fit[1]*binary_warped.shape[0] + left_fit[2]
    right_x_intercept = right_fit[0]*binary_warped.shape[0]**2 + right_fit[1]*binary_warped.shape[0] + right_fit[2]
    
    # Calculate the position of the vehicle
    position = (left_x_intercept + right_x_intercept) / 2
    
    # Calculate distance from center and convert to meters
    distance_from_center = abs(center - position) * xm_per_pix
    
    return distance_from_center

def main():
    ksize = (5, 5)
    canny_t1 = 50
    canny_t2 = 75
    thresh1 = 125
    thresh2 = 125

    rho = 1
    theta = np.pi/180
    hough_t = 30
    min_line_length = 10
    max_line_gap = 5
    x11 = 100
    
    calibration = np.load('output/cameraCalibration/calibration.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    img = cv.imread('test_images/test6.jpg')
    
    assert img is not None, "file could not be read, check with os.path.exists()"
    height, width, channels = img.shape
    # Print the shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    
    img_size = (img.shape[1], img.shape[0])
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

    undistorted = undistortImage(img, mtx, dist)
    hsvImage = hsv(undistorted)
    gauss = gaussian(hsvImage)
    warped =  warpImage(gauss,src,dst)
    gray = grayscale(undistorted) 
    thresh = threshold(warped,thresh1,thresh2)[1]
    canny = cannyTransformation(thresh,canny_t1,canny_t2)
    lines = houghLines(canny, rho, theta, hough_t, min_line_length, max_line_gap)
    image_with_lines = drawLines(img, lines)
    # image_with_lines = overlayLines(img, lines)
    cv.imshow('Canny Image', canny)
    cv.waitKey(0)
   # Assuming you have the binary_warped image from your perspective transform
    left_fit, right_fit = fit_polynomial(canny)
    
    if left_fit is None or right_fit is None:
        print("No lane pixels found.")
        return
    
    # Measure radius of curvature
    # left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit)
    left_curverad, right_curverad = measure_curvature_real(canny, left_fit, right_fit)

    
    # Calculate vehicle position
    distance_from_center = vehicle_position(left_fit, right_fit, canny)

    # Overlay text on the undistorted image
    cv.putText(undistorted, f'Left curvature radius: {left_curverad:.2f}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(undistorted, f'Right curvature radius: {right_curverad:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(undistorted, f'Distance from center: {distance_from_center:.2f}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image with overlaid text
    cv.imshow('Image with Overlay', undistorted)
    cv.waitKey(0)
    
    # Draw the lane lines on the image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(canny).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, canny.shape[0]-1, canny.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    Minv = cv.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    
    print("Undistorted image shape:", undistorted.shape)
    print("Newwarp image shape:", newwarp.shape)

    # Crop one row from the bottom and one column from the right of newwarp image
    newwarp_cropped = newwarp[:-1, :-1, :]
    # Combine the result with the original image
    result = cv.addWeighted(undistorted, 1, newwarp_cropped, 0.3, 0)
    
    cv.imshow('naziv', result)
    cv.waitKey(0)
    
main()




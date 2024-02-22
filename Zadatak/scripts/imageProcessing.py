import numpy as np
import cv2 as cv
import imageProcessingMethods as imageProcessingMethods
import laneDetection as laneDetection
import numpy as np
import cv2 as cv
import math as math
import glob as glob
import matplotlib.pyplot as plt
import os as os


"""Constant parameters"""
CANNY_T1 = 50
CANNY_T2 = 75
THRESHOLD_MIN = 100
THRESHOLD_MAX = 150
RHO = 1
THETA = np.pi/180
HOUGH_T = 30
MIN_LINE_GAP = 5
MAX_LINE_GAP = 10

def testImages():
 
    calibration = np.load('output/cameraCalibration/calibration.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    images = glob.glob('test_images/*.jpg')

    for fname in images:
        img = cv.imread(fname)        
        assert img is not None, "file could not be read, check with os.path.exists()"
        height, width, channels = img.shape

        print("Height:", height)
        print("Width:", width)
        print("Channels:", channels)
        
        if img.shape == (540, 960, 3):
            print("prvi if")
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

        undistortedImage = imageProcessingMethods.undistortImage(img, mtx, dist)
        hsvImage = imageProcessingMethods.hsv(undistortedImage)
        gaussImage = imageProcessingMethods.gaussian(hsvImage)
        warpedImage =  imageProcessingMethods.warpImage(gaussImage,src,dst)
        grayImage = imageProcessingMethods.grayscale(warpedImage) 
        thresholdImage = imageProcessingMethods.threshold(grayImage,THRESHOLD_MIN,THRESHOLD_MAX)[1]
        cannyImage = imageProcessingMethods.cannyTransformation(thresholdImage,CANNY_T1,CANNY_T2)
        lines = imageProcessingMethods.houghLines(cannyImage, RHO, THETA, HOUGH_T, MIN_LINE_GAP, MAX_LINE_GAP) 
        
        polygon_dst = dst.astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(warpedImage, [polygon_dst], isClosed=True, color=(0, 0, 255), thickness=2)
        cv.imshow('Original Image test', warpedImage)
        cv.waitKey(0)
        
        cv.imshow('Original Image', img)
        cv.waitKey(0)
        cv.imshow('HSV Image(warped)', hsvImage)
        cv.waitKey(0)
        cv.imshow('After threshold(warped)', thresholdImage)
        cv.waitKey(0)
        
        for points in lines:
            x1, y1, x2, y2 = np.array(points[0])
            cv.line(warpedImage, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            cv.imshow('HSV Image(warped) with Hough lines', warpedImage)
        cv.waitKey(0)

        # Assuming you have the binary_warped image from your perspective transform
        left_fit, right_fit = laneDetection.fit_polynomial(cannyImage)
        
        if left_fit is None or right_fit is None:
            print("No lane pixels found.")
            return None
        
        # Measure radius of curvature
        # left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit)
        left_curverad, right_curverad = laneDetection.measure_curvature_real(cannyImage, left_fit, right_fit)

        
        # Calculate vehicle position
        distance_from_center = laneDetection.vehicle_position(left_fit, right_fit, cannyImage)

        # Overlay text on the undistorted image
        cv.putText(undistortedImage, f'Left curvature radius: {left_curverad:.2f}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(undistortedImage, f'Right curvature radius: {right_curverad:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(undistortedImage, f'Distance from center: {distance_from_center:.2f}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw the lane lines on the image
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(cannyImage).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, cannyImage.shape[0]-1, cannyImage.shape[0])
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
        
        print("Undistorted image shape:", undistortedImage.shape)
        print("Newwarp image shape:", newwarp.shape)

        # Crop one row from the bottom and one column from the right of newwarp image
        newwarp_cropped = newwarp[:-1, :-1, :]
        # Combine the result with the original image
        result = cv.addWeighted(undistortedImage, 1, newwarp_cropped, 0.3, 0)
        
        cv.imshow('Final result', result)
        cv.waitKey(0)     
        
        output_dir = 'output/imageProcessing/'
        os.makedirs(output_dir, exist_ok=True)

        for i, processed_image in enumerate([undistortedImage, hsvImage, grayImage, warpedImage, result]):
            output_path = os.path.join(output_dir, f'{os.path.basename(fname)}_processed_{i}.jpg')
            cv.imwrite(output_path, processed_image)
            print(f"Processed image saved to: {output_path}")       




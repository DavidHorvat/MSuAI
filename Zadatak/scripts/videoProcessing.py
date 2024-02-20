import numpy as np
import cv2 as cv
import imageProcessingMethods as imageProcessingMethods
import laneDetection as laneDetection
import numpy as np
import cv2 as cv
import math as math
import glob as glob
import matplotlib.pyplot as plt

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

def testVideo():
    
    video_files = glob.glob('test_videos/*.mp4')
    for video_file in video_files:
        cap = cv.VideoCapture(video_file)
    
        calibration = np.load('output/cameraCalibration/calibration.npz')
        mtx = calibration['mtx']
        dist = calibration['dist']
        # cap = cv.VideoCapture('test_videos/project_video03.mp4')   
        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            assert img is not None, "file could not be read, check with os.path.exists()"
            
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

            undistortedFrame = imageProcessingMethods.undistortImage(img, mtx, dist)
            hsvFrame = imageProcessingMethods.hsv(undistortedFrame)
            gaussFrame = imageProcessingMethods.gaussian(hsvFrame)
            warpedFrame =  imageProcessingMethods.warpImage(gaussFrame,src,dst)
            grayFrame = imageProcessingMethods.grayscale(warpedFrame) 
            thresholdFrame = imageProcessingMethods.threshold(grayFrame,THRESHOLD_MIN,THRESHOLD_MAX)[1]
            cannyFrame = imageProcessingMethods.cannyTransformation(thresholdFrame,CANNY_T1,CANNY_T2)
            lines = imageProcessingMethods.houghLines(cannyFrame, RHO, THETA, HOUGH_T, MIN_LINE_GAP, MAX_LINE_GAP) 
            
        # Assuming you have the binary_warped image from your perspective transform
            left_fit, right_fit = laneDetection.fit_polynomial(cannyFrame)
            
            if left_fit is None or right_fit is None:
                print("No lane pixels found.")
                continue
            
            # Measure radius of curvature
            # left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit)
            left_curverad, right_curverad = laneDetection.measure_curvature_real(cannyFrame, left_fit, right_fit)
           
            # Calculate vehicle position
            distance_from_center = laneDetection.vehicle_position(left_fit, right_fit, cannyFrame)

            # Overlay text on the undistorted image
            cv.putText(undistortedFrame, f'Left curvature radius: {left_curverad:.2f}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(undistortedFrame, f'Right curvature radius: {right_curverad:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(undistortedFrame, f'Distance from center: {distance_from_center:.2f}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw the lane lines on the image
            # Create an image to draw the lines on
            warp_zero = np.zeros_like(cannyFrame).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            ploty = np.linspace(0, cannyFrame.shape[0]-1, cannyFrame.shape[0])
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
            
            # Crop one row from the bottom and one column from the right of newwarp image
            newwarp_cropped = newwarp[:-1, :-1, :]
            # Combine the result with the original image
            result = cv.addWeighted(undistortedFrame, 1, newwarp_cropped, 0.3, 0)
            
            cv.imshow('Final result', result)
            if cv.waitKey(1) == ord('q'):
                break
            
        cap.release()
        cv.destroyAllWindows()           


import numpy as np
import cv2 as cv

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
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Setting the necessary variables
numRows = 6 
numColumns = 9 
mean_error = 0

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((numRows*numColumns,3), np.float32)
objp[:,:2] = np.mgrid[0:numRows,0:numColumns].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('camera_cal/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (numRows,numColumns), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, (numRows,numColumns), corners2, ret)   
        cv.imshow('img', img)
        cv.waitKey(500)
    
# Perform calibration here
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez('output/cameraCalibration/calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
    
print( "total error: {}".format(mean_error/len(objpoints)) )

# Undistort the first image from the folder and save the pair as an image
fname = images[0]  # Selecting the first image
img = cv.imread(fname)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Ensure both images have the same height before concatenating
h_min = min(img.shape[0], dst.shape[0])
img = img[:h_min, :]
dst = dst[:h_min, :]

# Save the pair of images as a single image
pair_image = np.hstack((img, dst))
cv.imwrite('output/cameraCalibration/examplePair.png', pair_image)

# Undistort all images from the folder
for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Display the original and undistorted images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')

    # Plot undistorted image
    axes[1].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted Image')

    # Hide axes
    for ax in axes:
        ax.axis('off')

    plt.pause(2)  # Pause for 2 seconds before showing the next pair of images

    # Close the current figure to show the next pair of images
    plt.close(fig)

cv.destroyAllWindows()
    
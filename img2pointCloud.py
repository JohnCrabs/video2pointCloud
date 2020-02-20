from pointCloudCreation.img2pcl.calibrationCMR import *

import pandas as pd
from pyntcloud import PyntCloud

"""
import os
import cv2 as cv
import datetime as dt

These libraries are imported from pointCloudCreation.img2pcl.calibrationCMR
"""

SIFT_METHOD = 0
SURF_METHOD = 1
ORB_METHOD = 2


# -------------------------------------------------------------------------------------------------------------------- #
# 1) KeyPoint finder methods
# -------------------------------------------------------------------------------------------------------------------- #

def keyPointFinder(IMGpath: str, method=SIFT_METHOD):
    img = cv.imread(IMGpath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if method is SIFT_METHOD:
        methodExec = cv.xfeatures2d.SIFT_create()
    elif method is SURF_METHOD:
        methodExec = cv.xfeatures2d.SURF_create()
    elif method is ORB_METHOD:
        methodExec = cv.xfeatures2d.ORB_create()
    else:
        methodExec = cv.xfeatures2d.SIFT_create()
    kp, dest = methodExec.detectAndCompute(gray, None)

    return True, kp, dest


# -------------------------------------------------------------------------------------------------------------------- #
# 2) Create point cloud with calibrate images
# -------------------------------------------------------------------------------------------------------------------- #

def img2pcl_Calibrated(imgPathFolder: str, imgCalibFolderPath: str, chessboardSize=(7, 5), focalLength=None):
    if checkIfCalibrationNeeded(imgCalibFolderPath) is not True:
        chessboardCalibration(imgCalibFolderPath, chessboardSize)
    success, ret, mtx, dist, rvecs, tvecs, stereo_R, stereo_T = readCalibrationParameters(imgCalibFolderPath)

    # Take the path of all images in the given folder
    imageFiles = []
    fileCounter = 0
    print(str(dt.datetime.now()) + " : Reading files in folder")
    for r, d, f in os.walk(imgPathFolder):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
                    fileCounter += 1
    imageFiles.sort()
    # print(imageFiles) # For debugging

    img = cv.imread(imageFiles[0], 0)
    imgSize = img.shape[::-1]
    # Calculate Q

    if focalLength is None:
        success, Q = calculate_Q_without_f(imgSize, mtx, mtx, dist, dist, stereo_R, stereo_T)
    else:
        success, Q = calculate_Q_using_f(focalLength)

    print(str(dt.datetime.now()) + " : Create point cloud.")

    for i in range(1, len(imageFiles) - 1):  # len(imageFiles)
        success, disparity_map = disparityMap(imageFiles[i - 1], imageFiles[i], mtx=mtx, dist=dist)
        if success:
            success, points, colors = createPointCloud(imageFiles[i - 1], disparity_map, Q, mtx, dist)
            if success:
                print(str(dt.datetime.now()) + " : Export point cloud.")
                imgName = os.path.basename(imageFiles[i-1])
                exportPath = "outputData/cloud_" + os.path.splitext(imgName)[0] + ".ply"
                cloud = PyntCloud(pd.DataFrame(data=np.hstack((points, colors)),
                                               columns=["x", "y", "z", "red", "green", "blue"]))

                cloud.to_file(exportPath)

    return True


# -------------------------------------------------------------------------------------------------------------------- #
# 3) Create disparity map from 2 images
# -------------------------------------------------------------------------------------------------------------------- #

def disparityMap(imgPathL: str, imgPathR: str, mtx=None, dist=None):
    imgL_name = os.path.basename(imgPathL)
    imgR_name = os.path.basename(imgPathR)

    print(str(dt.datetime.now()) + " : Create disparity map for " + imgL_name + " and " + imgR_name)
    imgL = cv.imread(imgPathL)
    imgR = cv.imread(imgPathR)

    imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Take the size of the images. The two images must have the same size.
    h, w = imgL.shape[:2]
    imgSize = (w, h)
    # print(imgSize)  # for debugging

    if mtx is not None and dist is not None:
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imgSize, 1, imgSize)
        # Undistorted images
        imgL_gray = cv.undistort(imgL_gray, mtx, dist, None, new_camera_matrix)
        imgR_gray = cv.undistort(imgR_gray, mtx, dist, None, new_camera_matrix)

    # Set disparity parameters
    # Note: disparity range is tuned according to specific parameters obtained through trial and error.
    win_size = 5
    min_disp = 32
    max_disp = 112  # min_disp * 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=5,
                                  uniquenessRatio=5,
                                  speckleWindowSize=5,
                                  speckleRange=5,
                                  disp12MaxDiff=1,
                                  P1=8 * 3 * win_size ** 2,
                                  P2=32 * 3 * win_size ** 2)
    disparity_map = stereo.compute(imgL_gray, imgR_gray)
    # plt.imshow(disparity_map, 'gray') # for debugging
    # plt.show() # for debugging

    return True, disparity_map


# -------------------------------------------------------------------------------------------------------------------- #
# 4) Calculate Q
# -------------------------------------------------------------------------------------------------------------------- #

def calculate_Q_using_f(focalLength):
    focal = focalLength * 0.05

    Q = ([1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, focal, 0.0],
         [0.0, 0.0, 0.0, 1.0])

    Q = np.array(Q)

    return True, Q


def calculate_Q_without_f(imgSize, mtxL, mtxR, distL, distR, rvecs, tvecs):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(mtxL, distL, mtxR, distR, imgSize, rvecs, tvecs)

    return True, Q


# -------------------------------------------------------------------------------------------------------------------- #
# 4) Create point cloud
# -------------------------------------------------------------------------------------------------------------------- #

def createPointCloud(imgPath: str, disparity_map, Q, mtx=None, dist=None):
    img_name = os.path.basename(imgPath)

    print(str(dt.datetime.now()) + " : Create point cloud from image " + img_name)
    img = cv.imread(imgPath)

    # Take the size of the images. The two images must have the same size.
    h, w = img.shape[:2]
    imgSize = (w, h)
    # print(imgSize)  # for debugging

    if mtx is not None and dist is not None:
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imgSize, 1, imgSize)
        # Undistorted images
        img = cv.undistort(img, mtx, dist, None, new_camera_matrix)

    # Reproject points into 3D
    points_3D = cv.reprojectImageTo3D(disparity_map, Q)

    # Get color points
    colors = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    return True, output_points, output_colors

# -------------------------------------------------------------------------------------------------------------------- #
# 5) Create point cloud with non calibrate images
# -------------------------------------------------------------------------------------------------------------------- #

def img2pcl_Uncalibrated(imgPathFolder: str, focalLength=None, keyPoint_method=SIFT_METHOD):
    # Take the path of all images in the given folder
    imageFiles = []
    fileCounter = 0
    print(str(dt.datetime.now()) + " : Reading files in folder")
    for r, d, f in os.walk(imgPathFolder):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
                    fileCounter += 1
    imageFiles.sort()
    # print(imageFiles) # For debugging

    img = cv.imread(imageFiles[0], 0)
    imgSize = img.shape[::-1]
    # Calculate Q
    if focalLength is not None:
        success, Q = calculate_Q_using_f(focalLength)

    print(str(dt.datetime.now()) + " : Create point cloud.")

    for i in range(1, len(imageFiles) - 1):  # len(imageFiles)
        success, disparity_map = disparityMap(imageFiles[i - 1], imageFiles[i], mtx=mtx, dist=dist)
        if success:
            success, points, colors = createPointCloud(imageFiles[i - 1], disparity_map, Q, mtx, dist)
            if success:
                print(str(dt.datetime.now()) + " : Export point cloud.")
                imgName = os.path.basename(imageFiles[i-1])
                exportPath = "outputData/cloud_" + os.path.splitext(imgName)[0] + ".ply"
                cloud = PyntCloud(pd.DataFrame(data=np.hstack((points, colors)),
                                               columns=["x", "y", "z", "red", "green", "blue"]))

                cloud.to_file(exportPath)

    return True

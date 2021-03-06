from pointCloudCreation.img2pcl.calibrationCMR import *

"""
import os
import cv2 as cv
import datetime as dt

These libraries are imported from pointCloudCreation.img2pcl.calibrationCMR
"""

SIFT_METHOD = 0
SURF_METHOD = 1
ORB_METHOD = 2

FOCAL_APPROXIMATE = 0
FOCAL_CALCULATE = 1

# -------------------------------------------------------------------------------------------------------------------- #
# 0) Main Process Function
# -------------------------------------------------------------------------------------------------------------------- #

def img2pcl(imgFolderPath: str, exportPointCloudPath: str, focalLength=FOCAL_APPROXIMATE, imgCalibrationFolderPath=None,
            chessboardDimensions=(3, 3), exportCalibrationDataFolderPath=None, exportDisparityMapsFolderPath=None):
    # Check if there are Calibration Images (Create Point Cloud with Calibrated method)
    if imgCalibrationFolderPath is not None:
        if exportCalibrationDataFolderPath is None:
            exportCalibrationDataFolderPath = imgCalibrationFolderPath
        success = chessboardCalibration(imgCalibrationFolderPath, chessboardDimensions=chessboardDimensions,
                                        exportFilePath=exportCalibrationDataFolderPath)
        if not success:
            print(str(dt.datetime.now()) + " : Something went wrong with the Calibration Process")
            return False

    # Find FocalLength
    if focalLength is FOCAL_CALCULATE:
        findFocalLength(imgFolderPath, imgFolderPath)
    elif focalLength is FOCAL_APPROXIMATE:
        findFocalLengthApproximate(imgFolderPath, imgFolderPath)

    # Create Point Cloud
    createPointCloud(imgFolderPath, exportPointCloudPath, imgFolderPath,
                     calibrationFilesFolderPath=exportCalibrationDataFolderPath,
                     exportDisparityMapsFolderPath=exportDisparityMapsFolderPath)


# -------------------------------------------------------------------------------------------------------------------- #
# 1) Main Process Function
# -------------------------------------------------------------------------------------------------------------------- #

#Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def createPointCloud(imgFolderPath: str, exportPointCloudPath: str, focalLengthFolderPath: str,
                     calibrationFilesFolderPath=None, exportDisparityMapsFolderPath=None):
    # Calibration Parameters
    mtx = None
    dist = None
    stereo_R = None
    stereo_T = None
    cal_success = False

    # Take the path of all images in the given folder
    imageFiles = []
    fileCounter = 0
    print(str(dt.datetime.now()) + " : Reading files in folder")
    for r, d, f in os.walk(imgFolderPath):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
                    fileCounter += 1
    imageFiles.sort()
    # print(imageFiles)  # For debugging

    img = cv.imread(imageFiles[0], 0)
    imgSize = img.shape[::-1]

    # Read Calibration Parameters IF EXISTS
    if calibrationFilesFolderPath is not None:
        cal_success, ret, mtx, dist, rvecs, tvecs, stereo_R, stereo_T = readCalibrationParameters(
            calibrationFilesFolderPath)
        if not cal_success:
            print(str(dt.datetime.now()) + " : Calibration Parameter Files not Found.")
            return False

    # Read Focal Length IF EXIST
    # If not Check if Calibration Parameters EXISTS
    # If none of them exist then ERROR else Calculate Q with either Focal Length (primary way)
    #                                                        or Calibration Parameters (secondary way)
    success, focalLength = readFocalLength(focalLengthFolderPath)
    if not success:
        print(str(dt.datetime.now()) + " : Focal Length Not Found.")
        if cal_success is True:
            print(str(dt.datetime.now()) + " : Trying to Calculate Q with calibration parameters.")
            success, Q_mtrx = calculate_Q_without_f(imgSize, mtxL=mtx, mtxR=mtx, distL=dist, distR=dist,
                                                    rvecs=stereo_R, tvecs=stereo_T)
        else:
            print(str(dt.datetime.now()) + " : Cannot Calculate Q.")
            return False
    else:
        print(str(dt.datetime.now()) + " : Calculate Q Using Focal Length.")
        success, Q_mtrx = calculate_Q_using_f(focalLength)

    print(str(dt.datetime.now()) + " : Create point cloud.")

    for i in range(1, fileCounter):  # len(imageFiles)
        imgL = imageFiles[i - 1]
        imgR = imageFiles[i]
        imgL_name = os.path.basename(imgL)
        success, points, colors = createDisparityMapAndCloud(imgL, imgR,
                                                            exportDisparityMapsFolderPath=exportDisparityMapsFolderPath,
                                                             mtx=mtx, dist=dist, Q=Q_mtrx)
        # outPoints.append(points)
        # outColors.append(colors)

        # Define name for output file
        output_file = exportPointCloudPath + 'cloud_%s.ply' % os.path.splitext(imgL_name)[0]
        # Generate point cloud
        create_output(points, colors, output_file)

# -------------------------------------------------------------------------------------------------------------------- #
# 2) Calculate Q
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

    return True, np.array(Q)

# -------------------------------------------------------------------------------------------------------------------- #
# 3) Create disparity map from 2 images
# -------------------------------------------------------------------------------------------------------------------- #

def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


def createDisparityMapAndCloud(imgPathL: str, imgPathR: str, exportDisparityMapsFolderPath=None,
                               mtx=None, dist=None, Q=None):
    # Reading the image paths
    imgL_name = os.path.basename(imgPathL)
    imgR_name = os.path.basename(imgPathR)

    # Read Images
    print(str(dt.datetime.now()) + " : Create disparity map for " + imgL_name + " and " + imgR_name)
    imgL = cv.imread(imgPathL)
    imgR = cv.imread(imgPathR)

    # Take the size of the images. The two images must have the same size.
    h, w = imgL.shape[:2]
    imgSize = (w, h)
    # print(imgSize)  # for debugging

    if mtx is not None and dist is not None:
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imgSize, 1, imgSize)
        # Undistorted images
        imgL_undistored = cv.undistort(imgL, mtx, dist, None, new_camera_matrix)
        imgR_undistored = cv.undistort(imgR, mtx, dist, None, new_camera_matrix)
    else:
        imgL_undistored = imgL
        imgR_undistored = imgR
    # Downsample each image 3 times (because they're too big)
    imgL_downsampled = downsample_image(imgL_undistored, 3)
    imgR_downsampled = downsample_image(imgR_undistored, 3)

    # Set disparity parameters
    # Note: disparity range is tuned according to specific parameters obtained through trial and error.
    win_size = 7
    min_disp = -1
    max_disp = 15
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
    disparity_map = stereo.compute(imgL_downsampled, imgR_downsampled)
    # plt.imshow(disparity_map, 'gray') # for debugging
    # plt.show() # for debugging

    if exportDisparityMapsFolderPath is not None:
        exportDisparityMapName = "dispMap_" + os.path.splitext(imgL_name)[0] + "_" + os.path.splitext(imgR_name)[0]
        cv.imwrite(exportDisparityMapsFolderPath + exportDisparityMapName + ".jpg", disparity_map)

    # Reproject points into 3D
    # print(Q)
    points_3D = cv.reprojectImageTo3D(disparity_map, Q)

    # Get color points
    colors = cv.cvtColor(imgL_downsampled, cv.COLOR_BGR2RGB)

    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    return True, output_points, output_colors

# -------------------------------------------------------------------------------------------------------------------- #
# 4) KeyPoint finder methods
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

def stereoRectifyWithoutCalibration(imgL: str, imgR: str):

    success, kpL, destL = keyPointFinder(imgL)
    success, kpR, destR = keyPointFinder(imgR)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(destL, destR, k=2)

    good = []
    ptsL = []
    ptsR = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            ptsR.append(kpR[m.trainIdx].pt)
            ptsL.append(kpL[m.queryIdx].pt)

    ptsL = np.int32(ptsL)
    ptsR = np.int32(ptsR)
    F, mask = cv.findFundamentalMat(ptsL, ptsR, cv.FM_LMEDS)

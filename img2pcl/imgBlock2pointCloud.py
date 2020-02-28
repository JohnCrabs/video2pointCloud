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

H_MIN_SIZE = 2048
W_MIN_SIZE = 2048

DISP_CALC_WIN_SIZE = 15
DISP_CALC_MIN_DISP = 0
DISP_CALC_MAX_DISP = 32

# -------------------------------------------------------------------------------------------------------------------- #
# 0) Create cv::Mat Image List
# -------------------------------------------------------------------------------------------------------------------- #

def imgBlock2PointCloud(imgFolderPath: str, exportPointCloudPath: str, exportDisparityMapsFolderPath=None):
    imgPathList, imgNameList, imgSize_whc, imgListSize = createImagePathList(imgFolderPath)  # Take image information

    #imgVectorList = createImageVectorList(imgPathList)  # Create a list with images
    #imgVectorList, imgSize_whc = blockDownsample(imgVectorList, imgSize_whc)

    focalLength = approximateFocalLength(imgSize_whc[0])
    cam_mtrx = setCameraMatrix(fx=focalLength, fy=focalLength, cx=imgSize_whc[0]/2, cy=imgSize_whc[1]/2)

    # cam_mtrx = setCameraMatrix(fx=focalLength, fy=focalLength)
    success, Q_mtrx = calculate_Q_using_f(focalLength)

    #success, points, colors, cloudNum = createBlockDisparityMapAndCloud(imgVectorList, imgNameList, imgListSize,
    #                                                                    exportPointCloudPath, Q_mtrx,
    #                                                                    exportDisparityMapsFolderPath, cam_mtrx)

    success = createBlockDisparityMapAndCloud_FromList(imgPathList, imgNameList, imgListSize, exportPointCloudPath,
                                                       Q_mtrx, exportDisparityMapsFolderPath, cam_mtrx, imgSize_whc)

    return success

# -------------------------------------------------------------------------------------------------------------------- #
# 1) Create a list with all compatible image formats in a folder
# -------------------------------------------------------------------------------------------------------------------- #

def createImagePathList(srcFolder: str):
    imgPathList = []  # A list with the relative path of images
    imgNameList = []  # A list with the name and the extension of images
    imgNumber = 0     # The number of the supported images the folder contains

    print(str(dt.datetime.now()) + " : Reading files in folder.")

    # Find the imgPathList and imgNumber
    for r, d, f in os.walk(srcFolder):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imgPathList.append(os.path.join(r, file))
                    imgNumber += 1
    imgPathList.sort()

    # Take the name of images
    for file in imgPathList:
        imgNameList.append(os.path.basename(file))

    # Find the size of the first image.
    # All images needs to have the same size for the algorithm to run.
    img = cv.imread(imgPathList[0])
    imgSize_whc = img.shape
    imgSize_whc = [imgSize_whc[1], imgSize_whc[0], imgSize_whc[2]]

    return imgPathList, imgNameList, imgSize_whc, imgNumber

# -------------------------------------------------------------------------------------------------------------------- #
# 2) Create cv::Mat Image List
# -------------------------------------------------------------------------------------------------------------------- #

def createImageVectorList(srcList):
    imgVectorList = []
    print(str(dt.datetime.now()) + " : Create list from images.")
    for src in srcList:
        print(str(dt.datetime.now()) + " : Open : %s" % src)
        tmp = cv.imread(src)
        imgVectorList.append(np.array(tmp))

    return imgVectorList

# -------------------------------------------------------------------------------------------------------------------- #
# 3) Run Sift for block matching
# -------------------------------------------------------------------------------------------------------------------- #

def downsample_image(image, scaleFactor):
    for i in range(0, scaleFactor-1):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

def blockDownsample(imgVectorList, imgSize_whc):
    imgNewVectorList = []
    imgNewSize_whc = []
    print(str(dt.datetime.now()) + " : Downsample Image Block.")
    if imgSize_whc[0] > W_MIN_SIZE:
        scaleFactor = int(imgSize_whc[0]/W_MIN_SIZE)
        print(str(dt.datetime.now()) + " : Downsample Scale Factor = %d." % scaleFactor)
        for img in imgVectorList:
            imgNewVectorList.append(downsample_image(img, scaleFactor))
        imgNewSize_whc = imgNewVectorList[0].shape
        imgNewSize_whc = [imgNewSize_whc[1], imgNewSize_whc[0], imgNewSize_whc[2]]
    elif imgSize_whc[1] > H_MIN_SIZE:
        scaleFactor = int(imgSize_whc[1] / H_MIN_SIZE)
        print(str(dt.datetime.now()) + " : Downsample Scale Factor = %d." % scaleFactor)
        for img in imgVectorList:
            imgNewVectorList.append(downsample_image(img, scaleFactor))
        imgNewSize_whc = imgNewVectorList[0].shape
        imgNewSize_whc = [imgNewSize_whc[1], imgNewSize_whc[0], imgNewSize_whc[2]]

    return imgNewVectorList, imgNewSize_whc

def imgDownsample(image, imgSize_whc):
    print(str(dt.datetime.now()) + " : Downsample Image.")
    image_downsample = image
    imgNewSize_whc = imgSize_whc
    if imgSize_whc[0] > W_MIN_SIZE:
        scaleFactor = int(imgSize_whc[0] / W_MIN_SIZE)
        print(str(dt.datetime.now()) + " : Downsample Scale Factor = %d." % scaleFactor)
        image_downsample = downsample_image(image, scaleFactor)

    elif imgSize_whc[1] > H_MIN_SIZE:
        scaleFactor = int(imgSize_whc[1] / H_MIN_SIZE)
        print(str(dt.datetime.now()) + " : Downsample Scale Factor = %d." % scaleFactor)
        image_downsample = downsample_image(image, scaleFactor)
        imgNewSize_whc = imgNewVectorList[0].shape
        imgNewSize_whc = [imgNewSize_whc[1], imgNewSize_whc[0], imgNewSize_whc[2]]

    return image_downsample, imgNewSize_whc
# -------------------------------------------------------------------------------------------------------------------- #
# 4) Run Sift for block matching
# -------------------------------------------------------------------------------------------------------------------- #

def blockMatchingWith_SIFT(imgVectorList, imgNameList, imgListSize):
    print(str(dt.datetime.now()) + " : Run SIFT:")
    # Transform imgVectorList to grayScale
    imgGray = []
    for img in imgVectorList:
        imgGray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    sift = cv.xfeatures2d.SIFT_create()  # Set Executable method to SIFT

    print(str(dt.datetime.now()) + " : Set Matching Parameters.")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    F_mtrx = []
    F_size = 0
    counter = 1
    for img in imgGray:
        for index in range(counter, imgListSize):
            print(str(dt.datetime.now()) + " : Perform matching for %s " % imgNameList[counter-1] + "and "
                  + "%s " % imgNameList[index])

            kpL, descrL = sift.detectAndCompute(img, None)
            kpR, descrR = sift.detectAndCompute(imgGray[index], None)
            matches = flann.knnMatch(descrL, descrR, k=2)

            good = []
            ptsR = []
            ptsL = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.8 * n.distance:
                    good.append(m)
                    ptsR.append(kpR[m.trainIdx].pt)
                    ptsL.append(kpL[m.queryIdx].pt)

            ptsL = np.int32(ptsL)
            ptsR = np.int32(ptsR)
            F, mask = cv.findFundamentalMat(ptsL, ptsR, cv.FM_LMEDS)

            F_mtrx.append(F)
            F_size += 1
        counter += 1

    return F_mtrx, F_size

# -------------------------------------------------------------------------------------------------------------------- #
# 5) Approximate focal length and set camera matrix
# -------------------------------------------------------------------------------------------------------------------- #

def approximateFocalLength(imgWidth):
    focalLength = ((0.7*imgWidth) + imgWidth) / 2
    return focalLength

def setCameraMatrix(fx=1.0, fy=1.0, cx=0.0, cy=0.0):
    mtrx = ([fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1])

    mtrx = np.array(mtrx)

    return mtrx

def calculate_Q_using_f(focalLength):
    focal = focalLength * 0.005

    Q = ([1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, focal, 0.0],
         [0.0, 0.0, 0.0, 1.0])

    Q = np.array(Q)

    return True, Q

# -------------------------------------------------------------------------------------------------------------------- #
# 6) Create Disparity Maps and Point Cloud
# -------------------------------------------------------------------------------------------------------------------- #

def createBlockDisparityMapAndCloud_FromList(imgPathList, imgNameList, imgListSize, exportPointCloudPath, Q_mtrx,
                                             exportDisparityMapsFolderPath=None, cam_mtrx=None, imgSize_whc=None):
    #  Set up parameters
    win_size = DISP_CALC_WIN_SIZE
    min_disp = DISP_CALC_MIN_DISP
    max_disp = DISP_CALC_MAX_DISP
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    left_matcher = cv.StereoSGBM_create(minDisparity=min_disp,
                                        numDisparities=num_disp,
                                        blockSize=5,
                                        uniquenessRatio=15,
                                        speckleWindowSize=5,
                                        speckleRange=2,
                                        disp12MaxDiff=1,
                                        preFilterCap=63,
                                        P1=8 * 3 * win_size ** 2,
                                        P2=32 * 3 * win_size ** 2)

    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    # visual_multiplier = 1.0

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    points = []
    colors = []
    cloudNum = 0
    counter = 1
    for imgIndex in range(counter, imgListSize):
        imgL_name = imgNameList[counter - 1]
        imgR_name = imgNameList[counter]
        print(str(dt.datetime.now()) + " : Create point cloud from %s " % imgL_name + "and " + "%s " % imgR_name)

        imgL = cv.imread(imgPathList[counter - 1])
        imgR = cv.imread(imgPathList[counter])

        imgSize = imgSize_whc
        if imgSize_whc is None:
            imgSize = imgL.shape

        imgL_downsample, imgSize = imgDownsample(imgL, imgSize)
        imgR_downsample, imgSize = imgDownsample(imgR, imgSize)
        imgSize = imgL_downsample.shape[:2]

        if cam_mtrx is not None:
            dist = np.array([0, 0, 0, 0])
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cam_mtrx, dist, imgSize, 1, imgSize)
            # Undistorted images
            imgL = cv.undistort(imgL_downsample, cam_mtrx, dist, None, new_camera_matrix)
            imgR = cv.undistort(imgR_downsample, cam_mtrx, dist, None, new_camera_matrix)

        disparity_map_L = left_matcher.compute(imgL, imgR)
        disparity_map_R = right_matcher.compute(imgR, imgL)

        disparity_map_L = np.int16(disparity_map_L)
        disparity_map_R = np.int16(disparity_map_R)

        disparity_map = wls_filter.filter(disparity_map_L, imgL, None, disparity_map_R)
        disparity_map = cv.normalize(src=disparity_map, dst=disparity_map,
                                     beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        disparity_map = np.uint8(disparity_map)

        if exportDisparityMapsFolderPath is not None:
            exportDisparityMapName = "dispMap_" + os.path.splitext(imgL_name)[0] + "_" + \
                                     os.path.splitext(imgR_name)[0]
            cv.imwrite(exportDisparityMapsFolderPath + exportDisparityMapName + ".jpg", disparity_map)

        # Reproject points into 3D
        points_3D = cv.reprojectImageTo3D(disparity_map, Q_mtrx)

        # Get color points
        colorsL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)

        # Get rid of points with value 0 (i.e no depth)
        mask_map = disparity_map > disparity_map.min()

        # Mask colors and points.
        output_points = points_3D[mask_map] * 10
        output_colors = colorsL[mask_map]

        # Export point cloud
        output_file = exportPointCloudPath + 'cloud_%s_' % os.path.splitext(imgL_name)[0] + \
                      '%s.ply' % os.path.splitext(imgR_name)[0]
        create_output(output_points, output_colors, output_file)

        points.append(output_points)
        colors.append(output_colors)
        cloudNum += 1
        counter += 1
        print("\n")

    print(str(dt.datetime.now()) + " : Merge and Export PointCloud.")
    pointCloud = np.concatenate(points, axis=0)
    colorCloud = np.concatenate(colors, axis=0)
    output_file = exportPointCloudPath + '0000_cloud_All.ply'
    create_output(pointCloud, colorCloud, output_file)

    return True


def createBlockDisparityMapAndCloud(imgVectorList, imgNameList, imgListSize, exportPointCloudPath, Q_mtrx,
                                    exportDisparityMapsFolderPath=None, cam_mtrx=None):

    #  Set up parameters
    win_size = DISP_CALC_WIN_SIZE
    min_disp = DISP_CALC_MIN_DISP
    max_disp = DISP_CALC_MAX_DISP
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    left_matcher = cv.StereoSGBM_create(minDisparity=min_disp,
                                        numDisparities=num_disp,
                                        blockSize=5,
                                        uniquenessRatio=15,
                                        speckleWindowSize=5,
                                        speckleRange=2,
                                        disp12MaxDiff=1,
                                        preFilterCap=63,
                                        P1=8 * 3 * win_size ** 2,
                                        P2=32 * 3 * win_size ** 2)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    points = []
    colors = []
    cloudNum = 0
    counter = 1
    while counter < imgListSize:
        imgL_name = imgNameList[counter - 1]
        imgR_name = imgNameList[counter]
        print(str(dt.datetime.now()) + " : Create point cloud from %s " % imgL_name + "and " + "%s " % imgR_name)

        imgL = imgVectorList[counter-1]
        imgR = imgVectorList[counter]

        if cam_mtrx is not None:
            dist = np.array([0, 0, 0, 0])
            imgSize = imgL.shape[:2]
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cam_mtrx, dist, imgSize, 1, imgSize)
            # Undistorted images
            imgL = cv.undistort(imgL, cam_mtrx, dist, None, new_camera_matrix)
            imgR = cv.undistort(imgR, cam_mtrx, dist, None, new_camera_matrix)

        disparity_map_L = left_matcher.compute(imgL, imgR)
        disparity_map_R = right_matcher.compute(imgR, imgL)

        disparity_map_L = np.int16(disparity_map_L)
        disparity_map_R = np.int16(disparity_map_R)

        disparity_map = wls_filter.filter(disparity_map_L, imgL, None, disparity_map_R)
        disparity_map = cv.normalize(src=disparity_map, dst=disparity_map,
                                         beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        disparity_map = np.uint8(disparity_map)

        if exportDisparityMapsFolderPath is not None:
            exportDisparityMapName = "dispMap_" + os.path.splitext(imgL_name)[0] + "_" + \
                                     os.path.splitext(imgR_name)[0]
            cv.imwrite(exportDisparityMapsFolderPath + exportDisparityMapName + ".jpg", disparity_map)

        # Reproject points into 3D
        points_3D = cv.reprojectImageTo3D(disparity_map, Q_mtrx)

        # Get color points
        colorsL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)

        # Get rid of points with value 0 (i.e no depth)
        mask_map = disparity_map > disparity_map.min()

        # Mask colors and points.
        output_points = points_3D[mask_map] * 10
        output_colors = colorsL[mask_map]

        # Export point cloud
        output_file = exportPointCloudPath + 'cloud_%s_' % os.path.splitext(imgL_name)[0] + \
                      '%s.ply' % os.path.splitext(imgR_name)[0]
        create_output(output_points, output_colors, output_file)

        points.append(output_points)
        colors.append(output_colors)
        cloudNum += 1
        counter += 1

    pointCloud = np.concatenate(points, axis=0)
    colorCloud = np.concatenate(colors, axis=0)
    output_file = exportPointCloudPath + '0000_cloud_All.ply'
    create_output(pointCloud, colorCloud, output_file)

    return True, points, colors, cloudNum

def createBlockDisparityMapAndCloud_AllWithAll(imgVectorList, imgNameList, imgListSize, exportPointCloudPath, Q_mtrx,
                                    exportDisparityMapsFolderPath=None, cam_mtrx=None):

    #  Set up parameters
    win_size = DISP_CALC_WIN_SIZE
    min_disp = DISP_CALC_MIN_DISP
    max_disp = DISP_CALC_MAX_DISP
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    left_matcher = cv.StereoSGBM_create(minDisparity=min_disp,
                                        numDisparities=num_disp,
                                        blockSize=5,
                                        uniquenessRatio=15,
                                        speckleWindowSize=5,
                                        speckleRange=2,
                                        disp12MaxDiff=1,
                                        preFilterCap=63,
                                        P1=8 * 3 * win_size ** 2,
                                        P2=32 * 3 * win_size ** 2)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    points = []
    colors = []
    cloudNum = 0
    counter = 1
    for imgL_mat in imgVectorList:
        for imgR_index in range(counter, imgListSize):
            imgL_name = imgNameList[counter - 1]
            imgR_name = imgNameList[imgR_index]
            print(str(dt.datetime.now()) + " : Create point cloud from %s " % imgL_name + "and " + "%s " % imgR_name)

            imgL = imgL_mat
            imgR = imgVectorList[imgR_index]

            if cam_mtrx is not None:
                dist = np.array([0, 0, 0, 0])
                imgSize = imgL.shape[:2]
                new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cam_mtrx, dist, imgSize, 1, imgSize)
                # Undistorted images
                imgL = cv.undistort(imgL, cam_mtrx, dist, None, new_camera_matrix)
                imgR = cv.undistort(imgR, cam_mtrx, dist, None, new_camera_matrix)

            disparity_map_L = left_matcher.compute(imgL, imgR)
            disparity_map_R = right_matcher.compute(imgR, imgL)

            disparity_map_L = np.int16(disparity_map_L)
            disparity_map_R = np.int16(disparity_map_R)

            disparity_map = wls_filter.filter(disparity_map_L, imgL, None, disparity_map_R)
            disparity_map = cv.normalize(src=disparity_map, dst=disparity_map,
                                         beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
            disparity_map = np.uint8(disparity_map)

            if exportDisparityMapsFolderPath is not None:
                exportDisparityMapName = "dispMap_" + os.path.splitext(imgL_name)[0] + "_" + \
                                         os.path.splitext(imgR_name)[0]
                cv.imwrite(exportDisparityMapsFolderPath + exportDisparityMapName + ".jpg", disparity_map)

            # Reproject points into 3D
            points_3D = cv.reprojectImageTo3D(disparity_map, Q_mtrx)

            # Get color points
            colorsL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)

            # Get rid of points with value 0 (i.e no depth)
            mask_map = disparity_map > disparity_map.min()

            # Mask colors and points.
            output_points = points_3D[mask_map] * 10
            output_colors = colorsL[mask_map]

            # Export point cloud
            output_file = exportPointCloudPath + 'cloud_%s_' % os.path.splitext(imgL_name)[0] + \
                          '%s.ply' % os.path.splitext(imgR_name)[0]
            create_output(output_points, output_colors, output_file)

            points.append(output_points)
            colors.append(output_colors)
            cloudNum += 1
        counter += 1
    return True, points, colors, cloudNum

# -------------------------------------------------------------------------------------------------------------------- #
# 7) Export point cloud as .ply
# -------------------------------------------------------------------------------------------------------------------- #

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

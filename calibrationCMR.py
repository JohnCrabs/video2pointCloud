import os  # Use it for reading the paths
import cv2 as cv  # Key library for the calibration algorithm
import numpy as np  # Use it to save the parameters
import datetime as dt  # Use it for printing messages

#These libraries are needed for finding the focal length of an image
import PIL.ExifTags
import PIL.Image

imgFileFormats = (".jpg", ".jpeg", ".png", ".tiff")
calibFormat = ".npy"


def chessboardCalibration(folderIMGpath: str, chessboardDimensions=(3, 3), exportFilePath: str = None):
    """
    :param folderIMGpath: Folder with all calibration images.
    :param exportFilePath: The output file path in which all data will be stored.
    :param chessboardDimensions: The size of the chessboard pattern. Default 3x3
    :return: True/False
    """

    if checkForCalibrationFiles(exportFilePath):
        print(str(dt.datetime.now()) + " : Calibration files already exist in folder.")
        return True

    # Read the path of all images in a given folder
    imageFiles = []
    fileCounter = 0
    print(str(dt.datetime.now()) + " : Reading files in folder")
    for r, d, f in os.walk(folderIMGpath):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
                    fileCounter += 1
    imageFiles.sort()
    # print(imageFiles) # For debugging

    # Specify calibration parameters
    print(str(dt.datetime.now()) + " : Find calibration points for each image")
    terminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboardDimensions[0] * chessboardDimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardDimensions[0], 0:chessboardDimensions[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Calibrate the Camera
    procCounter = 1
    for fileName in imageFiles:
        # Print current process
        procIMG = os.path.basename(fileName)
        percentage = float(procCounter) / float(fileCounter) * 100
        print(str(dt.datetime.now()) + " : (%d" % procCounter + "/%d)" % fileCounter +
              "Process %s : " % procIMG + "%.2f%%" % percentage)
        procCounter += 1

        # Read Image and transform into grayscale
        img = cv.imread(fileName)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardDimensions, None)
        # print(ret)  # for debugging
        # print(corners)  # for debugging
        # cv.imwrite("outputData/tmp_%s.jpg" % procIMG, gray)  # for debugging

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), terminationCriteria)
            imgpoints.append(corners)
            # Draw and display the corners for debugging
            # img = cv.drawChessboardCorners(img, chessboardDimensions, corners, ret)  # for debugging
            # cv.imwrite("outputData/tmpChess_%s.jpg" % procIMG, img)  # for debugging

    print(str(dt.datetime.now()) + " : Calculate calibration parameter.")
    # Read Image and transform into grayscale
    img = cv.imread(imageFiles[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    retVal, cameraMtrxL, distCoeffsL, cameraMtrxR, distCoeffsR, R, T, E, F = cv.stereoCalibrate(objpoints,
                                                                                                imgpoints,
                                                                                                imgpoints,
                                                                                                mtx,
                                                                                                dist,
                                                                                                mtx,
                                                                                                dist,
                                                                                                gray.shape[::-1])
    print(str(dt.datetime.now()) + " : Export calibration parameters.")

    if exportFilePath is None:
        exportFilePath = folderIMGpath

    np.save(exportFilePath + "ret", ret)
    np.save(exportFilePath + "mtx", mtx)
    np.save(exportFilePath + "dist", dist)
    np.save(exportFilePath + "rvecs", rvecs)
    np.save(exportFilePath + "tvecs", tvecs)
    np.save(exportFilePath + "stereo_R", R)
    np.save(exportFilePath + "stereo_T", T)

    return True


# -------------------------------------------------------------------------------------------------------------------- #

def readCalibrationParameters(folderPath: str):

    # Check if all calibration files are in the folder
    if not checkForCalibrationFiles(folderPath):
        return False, None, None, None, None, None, None, None

    # Take the path of all numpy files in the given folder
    ret = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    stereo_R = None
    stereo_T = None
    paraFiles = []
    print(str(dt.datetime.now()) + " : Reading parameter files.")
    for r, d, f in os.walk(folderPath):
        for file in f:
            if calibFormat in file:
                paraFiles.append(os.path.join(r, file))
    paraFiles.sort()
    # print(paraFiles) # for debugging

    for file in paraFiles:
        if "dist" in file:
            dist = np.load(file)
        elif "mtx" in file:
            mtx = np.load(file)
        elif "ret" in file:
            ret = np.load(file)
        elif "rvecs" in file:
            rvecs = np.load(file)
        elif "tvecs" in file:
            tvecs = np.load(file)
        elif "stereo_R" in file:
            stereo_R = np.load(file)
        elif "stereo_T" in file:
            stereo_T = np.load(file)

    return True, ret, mtx, dist, rvecs, tvecs, stereo_R, stereo_T


# -------------------------------------------------------------------------------------------------------------------- #

def checkForCalibrationFiles(folderPath: str):
    # Take the path of all numpy files in the given folder
    paraFiles = []
    print(str(dt.datetime.now()) + " : Checking for existing calibration files.")
    for r, d, f in os.walk(folderPath):
        for file in f:
            if calibFormat in file:
                paraFiles.append(os.path.join(r, file))
    paraFiles.sort()
    # print(paraFiles)  # for debugging

    checkCounter = 0
    for file in paraFiles:
        if "dist" in file:
            print(str(dt.datetime.now()) + " : dist FILE OK.")
            checkCounter += 1
        elif "mtx" in file:
            print(str(dt.datetime.now()) + " : mtx FILE OK.")
            checkCounter += 1
        elif "ret" in file:
            print(str(dt.datetime.now()) + " : ret FILE OK.")
            checkCounter += 1
        elif "rvecs" in file:
            print(str(dt.datetime.now()) + " : rvecs FILE OK.")
            checkCounter += 1
        elif "tvecs" in file:
            print(str(dt.datetime.now()) + " : tvecs FILE OK.")
            checkCounter += 1
        elif "stereo_R" in file:
            print(str(dt.datetime.now()) + " : tvecs FILE OK.")
            checkCounter += 1
        elif "stereo_T" in file:
            print(str(dt.datetime.now()) + " : tvecs FILE OK.")
            checkCounter += 1

    if checkCounter is 7:
        print(str(dt.datetime.now()) + " : Calibration check finished successfully.")
        return True
    print(str(dt.datetime.now()) + " : I cant find needed calibration files.")
    return False

# -------------------------------------------------------------------------------------------------------------------- #

def findFocalLength(imgPath: str, exportFolderPath: str):
    imageFiles = []
    for r, d, f in os.walk(imgPath):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
    imageFiles.sort()
    # print(imageFiles)  # For debugging

    print(str(dt.datetime.now()) + " : Reading Focal Length from image %s" % os.path.basename(imageFiles[0]))
    # Get exif data in order to get focal length.
    exif_img = PIL.Image.open(imageFiles[0])

    exif_data = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif_img._getexif().items()
        if k in PIL.ExifTags.TAGS}

    # Get focal length in tuple form
    focal_length_exif = exif_data['FocalLength']

    # Get focal length in decimal form
    focal_length = focal_length_exif[0] / focal_length_exif[1]

    # Save focal length
    np.save(exportFolderPath + "focalLength", focal_length)
    print(focal_length)

    return True

# -------------------------------------------------------------------------------------------------------------------- #

def findFocalLengthApproximate(imgPath: str, exportFolderPath: str):
    imageFiles = []
    for r, d, f in os.walk(imgPath):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imageFiles.append(os.path.join(r, file))
    imageFiles.sort()
    # print(imageFiles)  # For debugging

    print(str(dt.datetime.now()) + " : Calculate Focal Length.")
    # Get exif data in order to get focal length.
    img = cv.imread(imageFiles[0], 0)
    h, w = img.shape[:2]

    # Get focal length in tuple form
    meanPixelSize = 0.046
    focal_length = (0.7*w + w)/2 * meanPixelSize

    # Save focal length
    np.save(exportFolderPath + "focalLength", focal_length)
    print(focal_length)

    return True

# -------------------------------------------------------------------------------------------------------------------- #

def readFocalLength(folderPath: str):
    # Take the path of all numpy files in the given folder
    focalLength = None
    paraFiles = []
    print(str(dt.datetime.now()) + " : Reading parameter files.")
    for r, d, f in os.walk(folderPath):
        for file in f:
            if calibFormat in file:
                paraFiles.append(os.path.join(r, file))
    paraFiles.sort()
    # print(paraFiles) # for debugging

    for file in paraFiles:
        if "focalLength" in file:
            focalLength = np.load(file)

    if focalLength is None:
        return False, None
    return True, focalLength

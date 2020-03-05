from img2pointCloud.calibrationCMR import *

import math as math

"""
import os
import cv2 as cv
import datetime as dt

These libraries are imported from pointCloudCreation.img2pcl.calibrationCMR
"""

H_MIN_SIZE = 2048
W_MIN_SIZE = 2048

AKAZE_METHOD = 0
ORB_METHOD = 1
SIFT_METHOD = 2
SURF_METHOD = 3

LOWE_RATIO = 0.9


# -------------------------------------------------------------------------------------------------------------------- #
# 0) Set Classes and Useful Functions
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------- #
# 0.c) Classes
# -------------------------------------------------------------- #
# -------------------------------------------------------------- #
# 0.c.col) Color
# -------------------------------------------------------------- #


class Color:
    red: int
    green: int
    blue: int
    gray: int
    is_colored: bool


# -------------------------------------------------------------- #
# 0.c.p2d) Point2D
# -------------------------------------------------------------- #


class Point2D:
    id: int
    x: float
    y: float


# -------------------------------------------------------------- #
# 0.c.p3d) Point3D
# -------------------------------------------------------------- #


class Point3D:
    x: float
    y: float
    z: float

    def point_info(self):
        print("(x, y, z) = ( ", self.x, ", ", self.y, ", ", self.z, " )")


# -------------------------------------------------------------- #
# 0.c.size) Size
# -------------------------------------------------------------- #


class Size:
    w: int
    h: int
    c: int

    def setSize(self, width, height, color_bands):
        self.w = width
        self.h = height
        self.c = color_bands


# -------------------------------------------------------------- #
# 0.c.cam) Camera
# -------------------------------------------------------------- #


class Camera:
    fx = 1.0
    fy = 1.0
    cx = 0.0
    cy = 0.0

    cam_matrix = []

    def approximateFocalLength(self, imgWidth, imgHeight):
        # Find the bigger size
        w = imgWidth
        if imgHeight > imgWidth:
            w = imgHeight
        focalLength = ((0.7 * w) + w) / 2  # Calculate the focal length as (70% of W + 100% of W) / 2 [px]
        self.fx = focalLength
        self.fy = focalLength

    def approximatePrincipalPoint(self, imgWidth, imgHeight):
        self.cx = imgWidth / 2
        self.cy = imgHeight / 2

    def setCameraParameters(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def approximateCameraParameters(self, size: Size):
        self.approximateFocalLength(imgWidth=size.w, imgHeight=size.h)
        self.approximatePrincipalPoint(imgWidth=size.w, imgHeight=size.h)

    def setCameraMatrix(self):
        mtrx = ([self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1])
        self.cam_matrix = np.array(mtrx)

    def cameraInfo(self):
        print("Camera:")
        print("fx = %d" % self.fx)
        print("fy = %d" % self.fy)
        print("cx = %d" % self.cx)
        print("cy = %d" % self.cy)
        print("cam_matrix = \n", self.cam_matrix)


# -------------------------------------------------------------- #
# 0.c.pose) PoseMatrix
# -------------------------------------------------------------- #


class PoseMatrix:
    T_mtrx: []

    def setStartingPoseMatrix(self):
        set_pose_mtrx = [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]]
        self.T_mtrx = np.array(set_pose_mtrx)

    def setPoseMtrx(self, T_mtrx):
        self.T_mtrx = np.array(T_mtrx)

    def setPoseMatrix_R_t(self, R, t):
        Rt = []
        Rt.append(R)
        Rt.append(t)
        Rt = np.concatenate(Rt, axis=1)

        poseMtrx = []
        poseMtrx.append(Rt)
        poseMtrx.append([[0.0, 0.0, 0.0, 1.0]])
        poseMtrx = np.concatenate(poseMtrx, axis=0)

        self.T_mtrx = np.array(poseMtrx)

    def take_R_and_t(self):
        R = self.T_mtrx[:3, :3]
        t = self.T_mtrx[:3, 3:]

        return R, t


# -------------------------------------------------------------- #
# 0.c.pose) ProjectionMatrix
# -------------------------------------------------------------- #


class ProjectionMatrix:
    P_mtrx: []

    def setStartingProjectionMatrix(self, cam_mtrx):
        projectionMtrx = []
        zeroMtrx = [[0], [0], [0]]
        projectionMtrx.append(cam_mtrx)
        projectionMtrx.append(zeroMtrx)
        projectionMtrx = np.concatenate(projectionMtrx, axis=1)
        self.P_mtrx = np.array(projectionMtrx)

    def setProjectionMatrix(self, cam_mtrx, R, t):
        Rt = self.setRtMatrix(R, t)
        proj_Mtrx = np.dot(cam_mtrx, Rt)
        self.P_mtrx = np.array(proj_Mtrx)

    def setProjectionMatrix_UsingPose(self, cam_mtrx, poseMtrx):
        R = poseMtrx[:3, :3]
        t = poseMtrx[:3, 3:]
        self.setProjectionMatrix(cam_mtrx, R, t)

    def setRtMatrix(self, R, t):
        Rt = []
        Rt.append(R)
        minus_Rt = np.dot(-R, t)
        Rt.append(minus_Rt)
        Rt = np.concatenate(Rt, axis=1)
        return Rt


# -------------------------------------------------------------- #
# 0.c.img) Image
# -------------------------------------------------------------- #


class Image:
    id = 0
    src = ""
    name = ""
    type = ""

    size = Size()
    camera = Camera()

    size_down = Size()
    camera_down = Camera()

    key_points = []
    descriptor = []

    def set_image(self, index: int, path: str, img_name: str, img_type: str, img_Size: Size):
        self.id = index
        self.src = path
        self.name = img_name
        self.type = img_type
        self.size = img_Size
        self.camera.approximateCameraParameters(size=self.size)
        self.camera.setCameraMatrix()

    def image_info(self):
        print("\n")
        print_message("Image Information:")
        print("id = %d" % self.id)
        print("path = %s" % self.src)
        print("name = %s" % self.name)
        print("type = %s" % self.type)
        print("size = ( %d" % self.size.w + " x %d" % self.size.h + " x %d )" % self.size.c)
        self.camera.cameraInfo()

    def set_keypoints_descriptors(self, keypoints, descriptors):
        self.key_points = keypoints
        self.descriptor = descriptors

    def set_downsample_parameters(self, size: Size):
        self.size_down.w = size.w
        self.size_down.h = size.h
        self.size_down.c = size.c
        self.camera_down.approximateCameraParameters(size=size)
        self.camera_down.setCameraMatrix()

# -------------------------------------------------------------- #
# 0.c.b_img) MatchImg
# -------------------------------------------------------------- #


class MatchImg:
    id: int

    imgL = Image()
    imgR = Image()

    match_points = []
    good_matches = []

    g_points_left = []
    g_points_left_ids = []

    g_points_right = []
    g_points_right_ids = []

    def set_id(self, match_id):
        self.id = match_id

    def setImages(self, imageLeft: Image, imageRight: Image):
        self.imgL = imageLeft
        self.imgR = imageRight

    def setMatches(self, matches):
        self.match_points = matches

    def setGoodMatches(self, matches):
        self.good_matches = matches

    def setGoodPointsLeft(self, goodPoints, goodPointsIndex):
        self.g_points_left = goodPoints
        self.g_points_left_ids = goodPointsIndex

    def setGoodPointsRight(self, goodPoints, goodPointsIndex):
        self.g_points_right = goodPoints
        self.g_points_right_ids = goodPointsIndex


# -------------------------------------------------------------- #
# 0.c.b_img) BlockImg
# -------------------------------------------------------------- #

class Landmark:
    id: int
    point3d = Point3D()
    img_seen = 0

    def set_landmark(self, index, p3d: Point3D, seen):
        self.id = index
        self.point3d = p3d
        self.img_seen = seen

    def add_another_pair(self, p3d: Point3D):
        self.point3d.x += p3d.x
        self.point3d.y += p3d.y
        self.point3d.z += p3d.z
        self.img_seen += 1

    def set_true_landmark(self):
        self.point3d.x /= self.img_seen - 1
        self.point3d.y /= self.img_seen - 1
        self.point3d.z /= self.img_seen - 1

    def take_avg_landmark(self):
        pt = Point3D
        pt.x = self.point3d.x / (self.img_seen - 1)
        pt.y = self.point3d.y / (self.img_seen - 1)
        pt.z = self.point3d.z / (self.img_seen - 1)

        return pt


# -------------------------------------------------------------- #
# 0.c.b_img) BlockImg
# -------------------------------------------------------------- #


class BlockImg:
    image = []
    matches = []
    landmark = []

    prev_pose_L = PoseMatrix()
    curr_pose_R = PoseMatrix()

    proj_mtrx_L = ProjectionMatrix()
    proj_mtrx_R = ProjectionMatrix()

    def append_new_image(self, img: Image):
        self.image.append(img)

    # ---------------------------------------------------------------------- #
    # 0.c.b_img.0) Create a list with all compatible image formats in a folder.
    # ---------------------------------------------------------------------- #

    def open_images_in_folder(self, path_f: str):

        print("\n")
        print_message("Reading files in folder.")
        imgPathList = []  # A list with the relative path of images

        # Find the imgPathList and imgNumber
        for r, d, f in os.walk(path_f):
            for imgFormat in imgFileFormats:
                for file in f:
                    if imgFormat in file:
                        imgPathList.append(os.path.join(r, file))
        imgPathList.sort()

        # Take the name of images
        counter_id = 0
        for file in imgPathList:
            img_name_type = os.path.basename(file)
            img_name = os.path.splitext(img_name_type)[0]
            img_type = os.path.splitext(img_name_type)[1]

            img_open = cv.imread(file)
            if img_open.size == 0:
                message = "Error: Cannot open image at %s" % file
                print_message(message)
            else:
                message = "Read image file at : %s" % file
                print_message(message)
                img_size_tmp = img_open.shape
                if len(img_size_tmp) is not 3:
                    img_size_tmp = [img_size_tmp[0], img_size_tmp[1], 1]
                img_size = Size()  # Create Size instance
                img_size.setSize(img_size_tmp[1], img_size_tmp[0], img_size_tmp[2])

                img = Image()  # Create image instance
                img.set_image(counter_id, file, img_name, img_type, img_size)

                self.image.append(img)
                self.image[counter_id].image_info()
                counter_id += 1

    # ---------------------------------------------------------------------- #
    # 0.c.b_img.1) Find KeyPoints and Descriptors for all Images.
    # ---------------------------------------------------------------------- #
    def findKeypoints(self, matchingMethod=AKAZE_METHOD):
        print("\n")
        print_message("Find Key-points and Descriptors.")

        for img in self.image:  # For each image in block
            imgRead = cv.imread(img.src, flags=cv.IMREAD_GRAYSCALE)  # Read the image
            imgRead, imgSize_tmp = imgDownsample(imgRead, img.size, False)  # Downscale the image and
            # don't show messages

            # Create new size and set it
            imgSize_down = Size()
            imgSize_down.setSize(imgSize_tmp[1], imgSize_tmp[0], img.size.c)
            img.set_downsample_parameters(imgSize_down)

            # img.camera_down.cameraInfo()  # Uncomment for debugging

            # Create key-point finder method checking the matchingMethod parameter (set by user)
            if matchingMethod is AKAZE_METHOD:
                method = cv.AKAZE_create()  # akaze method
            elif matchingMethod is ORB_METHOD:
                method = cv.ORB_create()  # orb method
            elif matchingMethod is SIFT_METHOD:
                method = cv.xfeatures2d.SIFT_create()  # sift method
            elif matchingMethod is SURF_METHOD:
                method = cv.xfeatures2d.SURF_create()  # surf method
            else:
                method = cv.AKAZE_create()  # if method checking failed use akaze method

            kp, descr = method.detectAndCompute(imgRead, None)  # detect and compute keypoints

            img.set_keypoints_descriptors(kp, descr)  # set key-points and descriptor per image

            message = " : Img_%s_keypoints = " % img.name + "%d" % len(img.key_points)
            print_message(message)

    # ---------------------------------------------------------------------- #
    # 0.c.b_img.2) Block Matching
    # ---------------------------------------------------------------------- #
    def blockMatching(self):
        print("\n")
        print_message("Block Matching:")

        # Set the start prev pose matrix for first image to default:
        self.prev_pose_L.setStartingPoseMatrix()

        # Find the Number of feature matching
        matchSize = 0
        for i in range(1, len(self.image)):
            matchSize += len(self.image) - i

        # Create matcher
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        # Create matches
        matchCounter = 1
        landmark_left_index = []
        for index_L in range(0, len(self.image) - 1):  # Take index for left image (src)
            for index_R in range(index_L + 1, len(self.image)):  # Take index for right image (dst)
                print("")
                message = "Match %d" % matchCounter + " out of " + "%d matches needed to perform." % matchSize
                print_message(message)

                match_tmp = MatchImg()  # Create temporary class match
                match_tmp.set_id(matchCounter - 1)  # Set the match pair id. The id starts from 0 and calculated as
                # (matchCounter - 1) = 1 - 1 = 0 for the first loop
                match_tmp.setImages(self.image[index_L], self.image[index_R])  # Set the images
                message = "Create match between %s" % match_tmp.imgL.name + " and " + "%s." % match_tmp.imgR.name
                print_message(message)

                # Find matches between images
                matchedPoints = matcher.knnMatch(match_tmp.imgL.descriptor, match_tmp.imgR.descriptor, k=2)
                match_tmp.setMatches(matchedPoints)  # Save the matches to class

                # Find all good points as per Lower's ratio.
                goodMatches = []
                pointsLeftImg = []
                pointsRightImg = []
                pointIndexLeftImg = []
                pointIndexRightImg = []
                for m, n in matchedPoints:
                    if m.distance < LOWE_RATIO * n.distance:
                        goodMatches.append(m)
                        pointsLeftImg.append(match_tmp.imgL.key_points[m.queryIdx].pt)  # Take p_coords for left img
                        pointsRightImg.append(match_tmp.imgR.key_points[m.trainIdx].pt)  # Take p_coords for right img

                        pointIndexLeftImg.append(m.queryIdx)
                        pointIndexRightImg.append(m.trainIdx)

                # Set lists to np.array()
                goodMatches = np.array(goodMatches)  # goodMatches = matches who passes Lowe's ratio test
                pointsLeftImg = np.array(pointsLeftImg)  # pointsLeftImg = coordinates (i,j) for left img
                pointsRightImg = np.array(pointsRightImg)  # pointsRightImg = coordinates (i,j) for right img

                pointIndexLeftImg = np.array(pointIndexLeftImg)  # pointIndexLeftImg = index for pointsLeftImg
                pointIndexRightImg = np.array(pointIndexRightImg)  # pointIndexRightImg = index for pointsRightImg

                match_tmp.setGoodMatches(goodMatches)  # Save the good matches
                match_tmp.setGoodPointsLeft(pointsLeftImg, pointIndexLeftImg)  # Save the good points left
                match_tmp.setGoodPointsRight(pointsRightImg, pointIndexRightImg)  # Save the good points left

                message = "Found good points %d" % len(goodMatches) + " out of %d matches." % len(matchedPoints)
                print_message(message)

                # Calculate Essential Matrix
                cam_mtrx = match_tmp.imgL.camera_down.cam_matrix
                E, mask = cv.findEssentialMat(pointsLeftImg, pointsRightImg, cam_mtrx, cv.RANSAC)

                # Calculate pose matrix R and t
                # poseVal = The number of pose points (we'll use these points to create the cloud)
                #    R    = Rotate Matrix
                #    t    = Translate Matrix
                #  mask   = Take values 0 and 1 and if 1 this point is pose point (object point)
                poseVal, R, t, mask = cv.recoverPose(E, pointsLeftImg, pointsRightImg, cam_mtrx)
                poseMask = mask  # Keep the mask in a variable poseMask (I done this for easier code reading)

                # Create a temporary pose mtrx for calculations
                pose_tmp = PoseMatrix()
                pose_tmp.setPoseMatrix_R_t(R, t)

                # Calculate the current pose matrix as: prev_pose_mtrx * pose_tmp_mtrx
                currPoseMtrx = np.dot(self.prev_pose_L.T_mtrx, pose_tmp.T_mtrx)
                self.curr_pose_R.setPoseMtrx(currPoseMtrx)  # Set current pose_mtrx as curr

                # Set projection matrices
                self.proj_mtrx_L.setProjectionMatrix_UsingPose(self.image[index_L].camera_down.cam_matrix,
                                                               self.prev_pose_L.T_mtrx)
                self.proj_mtrx_R.setProjectionMatrix_UsingPose(self.image[index_R].camera_down.cam_matrix,
                                                               self.curr_pose_R.T_mtrx)
                # Perform triangulation
                points4D = cv.triangulatePoints(self.proj_mtrx_L.P_mtrx, self.proj_mtrx_R.P_mtrx,
                                                np.transpose(pointsRightImg), np.transpose(pointsRightImg))

                # If there are more than 2 images
                kp_used = []  # This keypoints has been used in previous paring
                if match_tmp.id > 0:

                    for index_curr in match_tmp.g_points_left_ids:  # Read g_points_left ids now
                        same_found = False  # Create a boolean to check if there are the same kp is found in 2 pairs
                        for index_prev in self.matches[match_tmp.id - 1].g_points_left_ids:  # Read g_
                                                                                             # points_left ids prev
                            if index_curr == index_prev:  # if two index are same
                                same_found = True  # set boolean to True
                                break  # break the loop
                        if same_found:  # if boolean true
                            kp_used.append(index_curr)  # append the index
                        else:  # else
                            kp_used.append(-1)  # set -1 value, array indexes must be >= 0

                    # print(len(match_tmp.g_points_left_ids))
                    # print(len(self.matches[match_tmp.id - 1].g_points_left_ids))
                    # print(len(kp_used))

                    scale = 0.0  # Create scale value
                    scale_counter = 0  # Scale counter

                    prev_R, prev_t = self.prev_pose_L.take_R_and_t()  # Split previous pose matrix
                    camera_prev_xyz = prev_t  # cam_x = prev_t, cam_y = prev_t, cam_z = prev_t

                    # print(camera_prev_xyz)  # Uncomment for debugging

                    point3D_tmp = []
                    avg_landmark_tmp = []
                    landmark_index_tmp = []

                    for index in range(0, len(goodMatches)):  # for each point in goodMatches
                        kp_index = kp_used[index]  # Take the index in kp_used
                        if poseMask[index] and kp_index is not -1:  # if is masked and there is a landmark
                            for i in range(0, len(landmark_left_index)):  # find which landmark it is
                                #print(kp_index, landmark_left_index[i])
                                if kp_index is landmark_left_index[i]:
                                    #  Calculate coords for the new point
                                    p = Point3D()
                                    p.x = points4D[0][index] / points4D[3][index]
                                    p.y = points4D[1][index] / points4D[3][index]
                                    p.z = points4D[2][index] / points4D[3][index]

                                    #  Calculate coords for the old point
                                    avr_landmark_p = self.landmark[i].take_avg_landmark

                                    point3D_tmp.append(p)  # Set new point
                                    avg_landmark_tmp.append(avr_landmark_p)  # Set landmark curr average
                                    landmark_index_tmp.append(i)  # Set landmark indexing
                    '''
                    # Uncomment for debugging
                    for point in point3D_tmp:
                        point.point_info()
                    '''
                    # print(len(point3D_tmp))

                    for p1 in range(0, len(point3D_tmp)-1):
                        for p2 in range(p1+1, len(point3D_tmp)):
                            dp_1 = euclideanDist(point3D_tmp[p1], point3D_tmp[p2])
                            dp_2 = euclideanDist(avg_landmark_tmp[p1], avg_landmark_tmp[p2])

                            print(dp_1, " ", dp_2)

                for index in range(0, len(goodMatches)):
                    if poseMask[index]:
                        prev_id = match_tmp.g_points_left_ids[index]

                        p = Point3D()
                        landM = Landmark()

                        p.x = points4D[0][index] / points4D[3][index]
                        p.y = points4D[1][index] / points4D[3][index]
                        p.z = points4D[2][index] / points4D[3][index]

                        kp_index = -1
                        if len(kp_used) is not 0:
                            kp_index = kp_used[index]
                        if kp_index is not -1:
                            for i in range(0, len(landmark_left_index)):
                                if kp_index is landmark_left_index[i]:
                                    self.landmark[i].add_another_pair(landM.point3d)
                        else:
                            landM.set_landmark(len(self.landmark), p, 2)  # Create new landmark
                            self.landmark.append(landM)  # Append new landmark
                            #print(prev_id)
                            landmark_left_index.append(prev_id)

                # Append match_tmp to self.matches list
                self.matches.append(match_tmp)

                # In the end of the loop set current pose as prev pose and increase matchCounter by 1
                self.prev_pose_L.setPoseMtrx(self.curr_pose_R.T_mtrx)
                matchCounter += 1

        for index in range(0, len(self.landmark)):
            self.landmark[index].set_true_landmark()

# -------------------------------------------------------------- #
# 0.f) Useful Functions
# -------------------------------------------------------------- #
# -------------------------------------------------------------- #
# 0.f.messages) Print process information messages
# -------------------------------------------------------------- #


def print_message(message: str):
    print(str(dt.datetime.now()) + " : " + message)


# -------------------------------------------------------------- #
# 0.f.downsample) Downsample images using default img pixel
# -------------------------------------------------------------- #


def downsample(image, scaleFactor):
    for i in range(0, scaleFactor - 1):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


def imgDownsample(image, imgSize: Size, showMessages=True):
    if showMessages:
        print_message("Downsample message.")
    image_downsample = image

    # Check if width is greater than height
    if imgSize.w > imgSize.h:
        if imgSize.w > W_MIN_SIZE:
            scaleFactor = int(imgSize.w / W_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)

    # Check if height is greater than width
    elif imgSize.w < imgSize.h:
        if imgSize.h > H_MIN_SIZE:
            scaleFactor = int(imgSize.h / H_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)

    # If previous checks failed then image sizes are the same.
    # This branch is needed cause the program must check if the
    #      sizes are greater than default algorithm sizes.
    else:
        if imgSize.w > W_MIN_SIZE:
            scaleFactor = int(imgSize.w / W_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)
        elif imgSize.h > H_MIN_SIZE:
            scaleFactor = int(imgSize.h / H_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)
    imgNewSize = image_downsample.shape

    return image_downsample, imgNewSize


def euclideanDist(pt3d_1: Point3D, pt3d_2: Point3D):
    dx = pt3d_1.x - pt3d_2.x
    dy = pt3d_1.y - pt3d_2.y
    dz = pt3d_1.z - pt3d_2.z

    dist = dx*dx + dy*dy + dz*dz
    dist = math.sqrt(dist)

    return dist

# -------------------------------------------------------------------------------------------------------------------- #
# 1) CrabSFM : Core Function. Call this function to run Structure from Motion Algorithm
# -------------------------------------------------------------------------------------------------------------------- #


def CrabSFM(imgFolderPath: str, exportPointCloudPath: str, exportDisparityMapsFolderPath=None):
    blockImg = BlockImg()  # Create block
    blockImg.open_images_in_folder(imgFolderPath)  # Open images
    blockImg.findKeypoints()  # find key-points and descriptors
    blockImg.blockMatching()  # create block matching

    print(exportPointCloudPath, exportDisparityMapsFolderPath)
    return True


def fromHomogenous2Normal(X):
    """ Transform a point from homogenous to normal coordinates. """

    x = X[:-1, :]
    x /= X[-1, :]

    return x

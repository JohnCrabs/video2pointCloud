from pointCloudCreation.video2images import *
from pointCloudCreation.img2pcl.img2pointCloud import *

videoPath = "inputData/VideosInput/"
exportVideoFramesPath = "outputData/VideoFrames/"
imgPath = "inputData/CalibData/RedMi_Test/"
calibrationPath = "inputData/CalibData/CalibrationIMG_RedMi/"
exportCalibrationParameters = "outputData/CalibrationParameters/"
exportDisparityMaps = "outputData/DisparityMaps/"
exportPointCloud = "outputData/PointCloud/"

chessboardDim = (11, 8)

videoINfolder2image(videoPath, exportVideoFramesPath)

success = img2pcl(exportVideoFramesPath, exportPointCloud, exportDisparityMapsFolderPath=exportDisparityMaps)

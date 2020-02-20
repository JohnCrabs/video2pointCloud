from pointCloudCreation.img2pcl.img2pointCloud import *

pathVideo = "inputData/"
exportImgPath = "outputData/VideoFrames/"
calibrationPath = "inputData/CalibData/CalibrationIMG_RedMi/"

success = img2pcl_Calibrated(exportImgPath, calibrationPath, (11, 8))

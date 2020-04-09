import numpy as np
import cv2
import cv2.aruco as aruco
import pickle
from scipy.spatial import distance as dist
import process_common
import math


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])



def getImageCorners_singleMarker(inputImage, calibrationData):

    cameraMatrix, distorsionCoefficients, rvecs, tvecs = calibrationData

    # get corners
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    parameters.cornerRefinementWinSize = 2
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 50
    # parameters.minOtsuStdDev = 20

    h, w = inputImage.shape[:2]
    cameraMatrixNew, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distorsionCoefficients, (w, h), 0, (w, h))
    inputImage = cv2.undistort(inputImage, cameraMatrix, distorsionCoefficients, None, cameraMatrixNew)

    corners, ids, rejectedImqgPoints = aruco.detectMarkers(inputImage, aruco_dict, parameters=parameters)

    dst = aruco.drawDetectedMarkers(inputImage, corners)

    poseRvecs, poseTvecs, trash = aruco.estimatePoseSingleMarkers(corners, 0.04, cameraMatrix, distorsionCoefficients)

    if poseTvecs is not None and poseRvecs.shape[2] == 3:
        for i in range(len(poseTvecs)):
            aruco.drawAxis(dst, cameraMatrixNew, distorsionCoefficients, poseRvecs[i], poseTvecs[i], 0.1)

        from math import pi, atan2, asin

        R = cv2.Rodrigues(poseRvecs[0])[0]

        pitch  = atan2(-R[2][1], R[2][2])
        yaw = asin(R[2][0])
        roll   = atan2(-R[1][0], R[0][0])

        roll_angle = 180 * atan2(-R[2][1], R[2][2]) / pi
        pitch_angle = 180 * asin(R[2][0]) / pi
        yaw_angle = 180 * atan2(-R[1][0], R[0][0]) / pi

        cv2.imshow('complete', dst)

    if (len(corners) > 0):
        tlCorner = corners[0]

        ordered = process_common.order_points(tlCorner[0])

        pixelDistanceX = np.linalg.norm(ordered[0] - ordered[1])
        pixelDistanceY = np.linalg.norm(ordered[0] - ordered[3])

        tl = (ordered[0][0], ordered[0][1])

        nominal_size = 19
        d1 = 170 * pixelDistanceX / nominal_size
        d2 = 95 * pixelDistanceY / nominal_size

        tr = (int(tl[0] + d1*np.cos(roll)*np.cos(yaw)), int(tl[1] - d1*np.sin(roll)*np.cos(pitch)))
        bl = (int(tl[0] + d2*np.sin(roll)*np.cos(yaw)), int(tl[1] + d2*np.cos(roll)*np.cos(pitch)))
        tl = (int(tl[0]), int(tl[1]))

        br = ((bl[0] + tr[0]-tl[0]), (bl[1] + tr[1]-tl[1]))


        return (tl,tr,br,bl)




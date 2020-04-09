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



def getImageCorners_twoMarker(inputImage, calibrationData):

    cameraMatrix, distorsionCoefficients, rvecs, tvecs = calibrationData

    # get corners
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    parameters.cornerRefinementWinSize = 2
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 50

    h, w = inputImage.shape[:2]

    cameraMatrixNew, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distorsionCoefficients, (w, h), 0, (w, h))
    inputImage = cv2.undistort(inputImage, cameraMatrix, distorsionCoefficients, None, cameraMatrixNew)

    corners, ids, rejectedImqgPoints = aruco.detectMarkers(inputImage, aruco_dict, parameters=parameters)

    dst = aruco.drawDetectedMarkers(inputImage, corners)

    poseRvecs, poseTvecs, trash = aruco.estimatePoseSingleMarkers(corners, 0.02, cameraMatrix, distorsionCoefficients)

    if poseTvecs is not None and poseRvecs.shape[2] == 3:
        for i in range(len(poseTvecs)):
            aruco.drawAxis(dst, cameraMatrixNew, distorsionCoefficients, poseRvecs[i], poseTvecs[i], 0.1)

    if (len(corners) > 1):

        tlCorner = corners[0]
        brCorner = corners[1]

        tlOrdered = process_common.order_points(tlCorner[0])
        brOrdered = process_common.order_points(brCorner[0])

        if (tlOrdered[0][0] > brOrdered[0][0]):
            t = brOrdered
            brOrdered = tlOrdered
            tlOrdered = t

        pt = [0,0]

        sourcePoints = []
        destPoints = []

        for pt in tlOrdered:
            sourcePoints.append(pt)

        for pt in brOrdered:
            sourcePoints.append(pt)

        nominal_size = 20
        nominal_width = 132
        nominal_height = 104

        destPoints.extend([[0,0], [nominal_size,0], [nominal_size,nominal_size], [0,nominal_size]])
        destPoints.extend([[nominal_width, nominal_height], [nominal_width + nominal_size, nominal_height], [nominal_width + nominal_size, nominal_height + nominal_size], [nominal_width, nominal_height + nominal_size]])

        pts_src = np.array(sourcePoints)
        pts_dst = np.array(destPoints)

        pts_dst = pts_dst * (1200 / (nominal_width + nominal_size))


        h, status = cv2.findHomography(pts_src,pts_dst)
        newSize = (1200,int((1200/(nominal_width+nominal_size))*(nominal_height + nominal_size)))
        newSize = (newSize[0] + 50, newSize[1] + 50)
        offsetMatrix = [[1, 0,5], [0, 1, 5], [0, 0, 1]]
        h = np.dot(h,offsetMatrix)
        warped = cv2.warpPerspective(dst, h, newSize , None)
        cv2.imshow('preliminary', warped)



        return None




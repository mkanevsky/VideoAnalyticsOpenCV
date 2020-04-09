import numpy as np
import cv2
import cv2.aruco as aruco
import pickle
from scipy.spatial import distance as dist

from process_singleMarker import getImageCorners_singleMarker

cap = cv2.VideoCapture(0)
warped = None

#cameraMatrix, distorsionCoefficients, rvecs, tvecs
calibration_data = pickle.load( open( "calibration.pckl", "rb" ))

img = cv2.imread("snaps\\image_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('pre', gray)

corners = getImageCorners_singleMarker(img,calibration_data)



cv2.line(gray, corners[0], corners[1], (255, 0, 0), 2)
cv2.line(gray, corners[0], corners[3], (255, 0, 0), 2)
cv2.line(gray, corners[3], corners[2], (255, 0, 0), 2)
cv2.line(gray, corners[1], corners[2], (255, 0, 0), 2)

cv2.imshow('post', gray)

target_size = (500,300)

pts1 = np.float32([corners[0],corners[1],corners[2],corners[3]])
pts2 = np.float32([[0,0],[target_size[0],0],[target_size[0],target_size[1]],[0,target_size[1]]])

M = cv2.getPerspectiveTransform(pts1,pts2)

warped = cv2.warpPerspective(img,M,target_size,None,cv2.INTER_LINEAR )

cv2.imshow('warped', warped)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
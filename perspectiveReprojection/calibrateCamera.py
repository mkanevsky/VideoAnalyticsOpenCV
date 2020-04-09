import cv2.aruco as aruco
import numpy as np
import time
import cv2
import pickle


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(6,9,.025,.0125,dictionary)
img = board.draw((200*3,200*3))

#Dump the calibration board to a file
cv2.imwrite('markers\\calibration_board_charuco.png',img)

cap = cv2.VideoCapture(0)

allCorners = []
allIds = []
decimator = 0

for i in range(100):

    # image capture
    ret,frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run bitmap marker detection
    res = cv2.aruco.detectMarkers(gray,dictionary)

    # if found, do corner interpolation
    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)

        numOfCorners = res2[0]
        cornerCoordinates = res2[1]
        cornerIds = res2[2]

        if cornerCoordinates is not None and cornerIds is not None and len(cornerCoordinates)>3 and decimator%2==0:
            allCorners.append(cornerCoordinates)
            allIds.append(cornerIds)

        # draw markers
        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

    #show the image
    cv2.imshow('frame',gray)

    # loop until elapsed enough images, or 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    decimator+=1


#Try calibrating and save calibration data into a pickle
try:
    calibration, cameraMatrix, distorsionCoefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,gray.shape,None,None)
    print(cameraMatrix)

    f = open('calibration.pckl', 'wb')
    pickle.dump((cameraMatrix, distorsionCoefficients, rvecs, tvecs), f)
    f.close()

except:
    cap.release()

# clean up
cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import cv2.cv as cv
import time
import pickle

def nothing(*arg):
    pass


blobs = []

def process_video():
    cap = cv2.VideoCapture('hns.mp4')

    cv2.namedWindow('edge')
    cv2.createTrackbar('thresh1', 'edge', 230, 5000, nothing)
    cv2.createTrackbar('thresh2', 'edge', 3000, 5000, nothing)

    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh1 = cv2.getTrackbarPos('thresh1', 'edge')
        thresh2 = cv2.getTrackbarPos('thresh2', 'edge')
        edge = cv2.Canny(gray, thresh1, thresh2, apertureSize=5)

        contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 1. filter by area, perimeter
        contours = [c for c in contours if 20 <= cv2.contourArea(c) <= 5000] 
        #contours = [c for c in contours if 5 <= cv2.arcLength(c, True) <= 200] 
        
        # 2. blobs
        b = []
        for c in contours:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            b.append([cx, cy, cv2.contourArea(c)])
        
        blobs.append(b)

        cv2.drawContours(gray, contours, -1, (255,0,0), 3)
        cv2.imshow('edge', gray)
        #time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            while 1:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    break
        
        pickle.dump(blobs, open('blobs.pkl', 'wb'))

    cap.release()
    cv2.destroyAllWindows()

process_video()

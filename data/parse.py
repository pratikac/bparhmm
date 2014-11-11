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
    cv2.createTrackbar('thresh', 'edge', 130, 255, nothing)

    while (cap.isOpened()):
        try:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.getTrackbarPos('thresh', 'edge')
            ret, edge = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY) 

            contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # 1. filter by area
            #areas = [cv2.contourArea(c) for c in contours]
            #print areas
            contours = [c for c in contours if 100 <= cv2.contourArea(c) <= 800] 
            #contours = [c for c in contours if 5 <= cv2.arcLength(c, True) <= 200] 
            
            # 2. blobs
            b = []
            for c in contours:
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                b.append([cx, cy, cv2.contourArea(c)])
            print b 
            blobs.append(b)

            cv2.drawContours(edge, contours, -1, (255,0,0), 3)
            cv2.imshow('edge', edge)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except:
            break

    pickle.dump(blobs, open('blobs.pkl', 'wb'))
    cap.release()
    cv2.destroyAllWindows()

process_video()

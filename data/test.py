import numpy as np
import cv2

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def on_mouse(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(first_frame,(ix,iy),(x,y),(0,255,255), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(first_frame,(ix,iy),(x,y),(0,0,255), 1)
        rois.append(((ix,iy),(x,y)))

rois = []
cap = cv2.VideoCapture('hns.mp4')
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_mouse)

ret, first_frame = cap.read()
cv2.setMouseCallback('frame', on_mouse)

while 1:
    cv2.imshow('frame', first_frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        break

id = 0
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.1, 10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(gray, (x,y), 3, 255, -1)
    

    cv2.imshow('frame', gray)
    

    id += 1
    if id > 1920:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import cv2.cv as cv
import time
import pickle
from pykalman import KalmanFilter



def process_video():
    blobsT = []
    cap = cv2.VideoCapture('hns.mp4')

    def nothing(*arg):
        pass
    cv2.namedWindow('edge')
    cv2.createTrackbar('thresh', 'edge', 130, 255, nothing)

    def get_blobs(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.getTrackbarPos('thresh', 'edge')
        ret, edge = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
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
            b.append([cx, cy])
        return b, contours, edge

    def dist(a,b):
        return np.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2)

    ret, frame = cap.read()
    blobs,contours,edge = get_blobs(frame)
    if len(blobs) < 3:
        raise
    curr_blobs = blobs[:]
    curr_blobs_covar = [1e-4*np.eye(2),1e-4*np.eye(2),1e-4*np.eye(2)]
    eps = 30
    
    kf = [None for i in xrange(3)]
    for i in xrange(3):
        kf[i] = KalmanFilter(initial_state_mean = curr_blobs[i], initial_state_covariance = 1e-4*np.eye(2),\
            transition_matrices = [[1,0],[0,1]], observation_matrices = [[1,0],[0,1]],\
            transition_covariance = 1e-1*np.eye(2), observation_covariance = 1e-6*np.eye(2))
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        blobs,contours,edge = get_blobs(frame)
        
        new_blobs = ['n','n','n']
        for b in blobs:
            ds = [dist(b, curr_blobs[i]) for i in xrange(3)]
            best_match_id = np.argmin(ds)
            if ds[best_match_id] < eps:
                new_blobs[best_match_id] = b
                
        for i in xrange(3):
            if new_blobs[i] != 'n':
                curr_blobs[i], curr_blobs_covar[i] = kf[i].filter_update(curr_blobs[i], curr_blobs_covar[i], new_blobs[i])
            else:
                curr_blobs[i], curr_blobs_covar[i] = kf[i].filter_update(curr_blobs[i], curr_blobs_covar[i], None)
            print new_blobs[i], curr_blobs[i]
            cv2.circle(edge, (int(curr_blobs[i][0]), int(curr_blobs[i][1])), 15, (0,255,0))

        blobsT.append(curr_blobs[:])

        cv2.drawContours(edge, contours, -1, (255,0,0), 2)
        cv2.imshow('edge', edge)
        #time.sleep(0.01)

        if (len(blobsT) > 1923) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    return blobsT

blobsT = np.array(process_video())
#pickle.dump(blobsT, open('blobs.pkl', 'wb'))

'''
blobsT = pickle.load(open('blobs.pkl','rb'))
blobsT = np.array(blobsT)
filtered_blobsT = []
for a in xrange(0,1):
    obs = np.ma.asarray(np.asarray(blobsT[:,a]))
    
    curr, curr_covar = obs[0], 1e-2*np.eye(2)
    for oi,o in enumerate(obs):
        curr_obs = None
        if len(o) == 2:
            curr_obs = o
        next_state, next_state_covar = kf.filter_update(curr, curr_covar, curr_obs)
        curr, curr_covar = next_state, next_state_covar
        print curr
'''

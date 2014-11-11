import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


blobsT = pickle.load(open('blobs.pkl','rb'))

T = len(blobsT)
N = 3
eps = 5

def dist(a,b):
    return np.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2 + 0.1*(a[2]-b[2])**2)

# 1. filter out repeated detections
def is_close(a,b,eps):
    if dist(a,b) < eps:
        return True
    else:
        return False

def is_far(a,b,eps):
    return not is_close(a,b,eps)

filtered_blobsT = []

curs = []
cur = blobsT[0]
filtered_blobsT.append(cur)
curs.append(cur)
for blobs in blobsT:
    cur = cur[:]
    new_blobs = [None,None,None]
    # only permuted
    for b in blobs:
        if is_close(b, cur[0], eps):
            if new_blobs[0]:
                print blobs, cur, new_blobs
                raise Exception
            new_blobs[0] = b
            cur[0] = b
        elif is_close(b, cur[1], eps):
            if new_blobs[1]:
                print blobs, cur, new_blobs
                raise Exception
            new_blobs[1] = b
            cur[1] = b
        elif is_close(b, cur[2], eps):
            if new_blobs[2]:
                print blobs, cur, new_blobs
                raise Exception
            new_blobs[2] = b
            cur[2] = b
        
    curs.append(cur)
    filtered_blobsT.append(new_blobs)

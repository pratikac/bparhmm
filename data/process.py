import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


blobsT = pickle.load(open('blobs.pkl','rb'))

T = len(blobsT)
N = 3
eps = 35

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


def find_next_seen_at(i,j):
    for ii in xrange(i+1, len(filtered_blobsT)):
        if filtered_blobsT[ii][j] != None:
            return ii
    return None

interpolated_blobsT = []
interpolated_blobsT.append(filtered_blobsT[0][:])

for i, blobs in enumerate(filtered_blobsT):
    if i == 0:
        continue
    interpolated_blobs = [[],[],[]]
    for j, b in enumerate(blobs):
        if b == None:
            last_seen = interpolated_blobsT[i-1][j]
            next_seen_at = find_next_seen_at(i,j)
            if next_seen_at != None:
                next_seen = filtered_blobsT[next_seen_at][j]
                t1 = np.array(last_seen)
                t2 = np.array(next_seen)
                interp = t1 + (t2-t1)/float(next_seen_at - (i-1))
                interpolated_blobs[j] = interp.tolist()
        else:
            interpolated_blobs[j] = b
    
    if [] in interpolated_blobs:
        print i, filtered_blobsT[i], interpolated_blobs
        raw_input()

    interpolated_blobsT.append(interpolated_blobs)

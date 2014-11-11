import cPickle as pickle
import numpy as np

blobs = pickle.load(open('blobs.pkl','rb'))

T = len(blobs)
N = 0


def dist(a,b):
    return np.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2)

# 1. filter out repeated detections
def is_close(a,b,eps):
    if dist(a,b) < eps:
        return True
    else:
        return False

def is_far(a,b,eps):
    return not is_close(a,b,eps)

def uniquefy(b):
    neighbors = [0 for bi in b]
    for i in xrange(len(b)):
        for j in xrange(i+1, len(b)):
            if is_close(b[i], b[j], 3):
                neighbors[i] = 1
                break
    return [b[i] for i in xrange(len(b)) if neighbors[i] == 0]

blobs = map(uniquefy, blobs)


# 2. filter locations of blobs
locs = []
for i in xrange(1, len(blobs)):
    blob = blobs[i]
    new_loc = []
    if not len(blob):
        locs.append(blob)
    else:
        loc = locs[-1]
        for b in blob:
            ds = [dist(l,b) for l in loc]
            if (not len(ds)) or min(ds) < 10 or min(ds) > 20:
                new_loc.append(b)
        locs.append(new_loc)
        print np.shape(new_loc)
    #print blob, new_loc

for li in xrange(len(locs)-1):
    loc1 = loc[li]
    loc2 = loc[li+1]
    for l1 in loc1:
        ds = [dist(l1,l2) for l2 in loc2]
        if min(ds) > 20:
            loc2.append(l1)
    
    print loc1

blobs = locs



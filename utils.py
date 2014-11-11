import numpy as np

def almost_equal(x,y,eps):
    '''
    x,y numpy 1d arrays
    returns true if |(x-y)|./|x| < eps
    '''
    d = np.divide(np.abs(x-y),np.abs(x))
    if max(d) > eps:
        return False
    return True

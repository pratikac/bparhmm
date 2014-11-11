import numpy as np


def forward_filter(P, softev, init):
    '''
    init: [K,1]
        initial distribution
    P: [K, K]
        P_ij = P(S_{t+1} = j | S_t = i)
    softev: [K,T]
        softev_{ij} = P(y(j) | s = i)
    alpha: [K, T]
        alpha_ij = prob. of sequences that end at i at time j
    '''
    P = np.transpose(P)
    K,c = np.shape(P)
    if not K == c:
        print 'P is not square'
    
    r, T = np.shape(softev)
    if not r == K:
        print 'softev wrong shape'
    
    scale = [0 for i in xrange(T)]
    alpha = np.zeros((K,T))
    alpha[:,0] = np.multiply(init, softev[:,0])
    scale[0] = np.sum(alpha[:,0])
    alpha[:,0] /= scale[0]
    
    for i in range(1, T):
        alpha[:,i] = np.multiply(np.dot(P, alpha[:,i-1]), softev[:,i])
        scale[i] = np.sum(alpha[:,i])
        #print softev[:,i], alpha[:,i], scale[i]
        alpha[:,i] /= scale[i]

    loglik = np.sum(np.log(scale))
    return alpha, loglik
    
def backward_smoothing(P, softev):
    '''
    P: [K, K]
        P_ij = P(S_{t+1} = j | S_t = i)
    softev: [K,T]
        softev_{ij} = P(y(j) | s = i)
    beta: [K, T]
        beta_ij = prob. of sequences that start i at time j
    '''
    K,c = np.shape(P)
    if not K == c:
        print 'P is not square'
    
    r, T = np.shape(softev)
    if not r == K:
        print 'softev wrong shape'
    
    PM = np.zeros((K,T))
    beta = np.zeros((K,T))
    beta[:,-1:] = np.ones((K,1))
    
    for i in reversed(range(T-1)):
        PM[:,i+1] = np.multiply(beta[:,i+1], softev[:,i+1])
        beta[:,i] = np.dot(P, PM[:,i+1])
        beta[:,i] = beta[:,i]/np.sum(beta[:,i])

    PM[:,0] = np.multiply(beta[:,0], softev[:,0]) 
    
    return beta, PM

def test_forward_backward():
    def p_y_s(yi, si):
        return np.exp(-0.5*(yi-si)**2)

    ''' 
    K,T = 2, 8
    P = np.array([[0.7, 0.3], [0.2, 0.8]])
    init = np.transpose(np.array([0.5, 0.5]))
    Y = [0,0,1,0,1,1,0,0]
    '''
    
    K,T = 2, 8
    P = np.random.rand(K,K)
    for i in xrange(K):
        P[i,:] = P[i,:]/np.sum(P[i,:])
    Y = np.random.randint(2, size=(T,)).tolist()
    init = np.transpose(np.array([1/float(K) for k in xrange(K)]))
    print P, Y, init

    softev = np.zeros((K,T))
    for t in xrange(T):
        for k in xrange(K):
            softev[k,t] = p_y_s(Y[t], k)
    
    print forward_filter(P, softev, init)
    print backward_smoothing(P, softev)

if __name__=='__main__':
    test_forward_backward()
    

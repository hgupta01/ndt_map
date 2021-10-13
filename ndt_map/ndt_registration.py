import re
import numpy as np
from math import sin, cos
from scipy.linalg import norm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class NDTRegistration:
    def __init__(self, meanF, covF):
        self.meanF_ = meanF
        self.covF_ = covF
        self.knn = NearestNeighbors(n_neighbors=1, radius=1.0)
        self.knn.fit(self.meanF_)
        
    def p2dCostFunc(self, rt, mF, cF, pM, knn=None):
        r, t = rt[0:3], rt[3:6]
        theta = norm(r)
        if ( theta == 0 ):
            R = np.eye(3)
            mMRnT = pM@R.T + t
        else:
            M = 1.0/theta*np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R = np.eye(3) + M*sin(theta) + M.dot(M)*(1 - cos(theta))
            mMRnT = pM@R.T + t

        nbrs = knn.radius_neighbors(mMRnT, 0.75, return_distance=False)
        
        score = 0
        for i in range(len(nbrs)):
            if len(nbrs[i])==0: continue
            mu = mMRnT[i]-mF[nbrs[i]]
            score = score - np.exp(-0.5*np.matmul(mu.reshape(-1,1,3),np.matmul(np.linalg.inv(cF[nbrs[i]]), mu.reshape(-1,3,1)))).sum()
        
        return score
    
    def d2dCostFunc(self, rt, mF, cF, mM, cM, knn=None):
        r, t = rt[0:3], rt[3:6]
        theta = norm(r)
        if ( theta == 0 ):
            R = np.eye(3)
            mMRnT = mM@R.T + t
        else:
            M = 1.0/theta*np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R = np.eye(3) + M*sin(theta) + M.dot(M)*(1 - cos(theta))
            mMRnT = mM@R.T + t

        nbrs = knn.radius_neighbors(mMRnT, 1.5, return_distance=False)
        
        score = 0
        for i in range(len(nbrs)):
            if len(nbrs[i])==0: continue
            mu = mMRnT[i]-mF[nbrs[i]]
            cov = R.T@cM[i]@R + cF[nbrs[i]]
            score = score + (-np.exp(-0.5*np.matmul(mu.reshape(-1,1,3),np.matmul(np.linalg.inv(cov), mu.reshape(-1,3,1))))).sum()
        return score
    
    def p2d(self, pM):
        rt = np.zeros(6)
        results = minimize(self.p2dCostFunc, rt, args = (self.meanF_, self.covF_, pM, self.knn), method = 'BFGS')
        return results.x
    
    def d2d(self, mM, cM):
        rt = np.zeros(6)
        results = minimize(self.d2dCostFunc, rt, args = (self.meanF_, self.covF_, mM, cM, self.knn), method = 'BFGS')
        return results.x
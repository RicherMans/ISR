'''
Created on Sep 15, 2014

@author: richman
'''
from GMM_Estimator import GMMEstimator
import numpy as np


def main():
    estimator = GMMEstimator(2)
    
#     radomly initialize 10000 samples of feature size 39
    samples = np.random.sample(size=(10000,39))
#     print np.random.multivariate_normal(mean=[0,1,3],cov=[[0,1,2],[1,0,2],[1,3,1]])
    estimator = estimator.fit(samples)
    print estimator.predict(samples)[:100]
    
    
if __name__ == '__main__':
    main()
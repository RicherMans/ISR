'''
Created on Nov 5, 2014

@author: richman
'''
from argparse import ArgumentParser
import numpy as np
import decompositions
import logging as log
import matplotlib.pyplot as plt 
import sklearn.lda

def main():
    log.basicConfig(filename='run.log',level=log.DEBUG)
    dims  = 3
    means = np.random.randint(-10,10,size=(dims))
    variances = np.random.sample(size=(dims,dims))
#     We sample 2 classes with 10 x,y coordinates each
    datapoints = np.random.multivariate_normal(means,variances,size=(2,10000)).T
#     datapoints= datapoints.reshape(2,150,4)
    log.debug("Inital datapoints shape:{0}".format(datapoints.shape))
    for datap,color in zip(datapoints,('red','blue','green')):
        plt.scatter(datap[:,0],datap[:,1],marker='+',c=color)
#     log.info("x:",x)
#     plt.plot(x,y,'x'); 
    plt.axis('equal');
    cl,samp,feat = datapoints.shape
    ldaproj= decompositions.lda(datapoints,1)
    
    datapoints = datapoints.reshape(samp*cl,feat)
    y = []
    for i in range(cl*samp):
        y.append(i/samp)
    ldask = sklearn.lda.LDA(1)
    ret = ldask.fit(datapoints,y).transform(datapoints)
    ret = ret.reshape(cl,samp)
#     TODO: Change target
    for i,marker,color in zip(
        range(dims),('^','+','o'),('red','blue','green')):
        plt.scatter(ldaproj[i],np.ones_like(ldaproj[i]),color=color,alpha=0.5,marker=marker)
        plt.scatter(ret[i], np.zeros_like(ret[i]), c=color)
#         plt.plot(ldaproj[i],np.zeros_like(ldaproj[i]),color=color,alpha=0.5)
#         plt.scatter(x=ldaproj[0],
#                 y=ldaproj[1],
#                 marker=marker,
#                 color=color,
#                 alpha=0.5,
#                 )
#     plt.scatter(x,marker='o',c='green')
#     plt.scatter(y[:,0],y[:,1],marker='o',c='red')
#     for c in range(len(ldaproj)):
#         plt.plot(ldaproj[c],'x')
    plt.show()
    
def parseArgs():
    parser = ArgumentParser()
    parser.add_argument()
    return parser.parse_args()
if __name__ == "__main__":
    main()
     

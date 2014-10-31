'''
Created on Sep 15, 2014

@author: richman
'''
from GMM_Estimator import GMMEstimator
import numpy as np
import argparse

def main():
    args = parseargs()
    gmms=[]
    if args.dev:
        devfile = args.dev.readlines()
        devfile = separateTrain(devfile)
    if args.train:
        trainfile=args.train.readlines()
        trainfile = separateTrain(trainfile)
    for train in trainfile.items():
        label,data = train
        gmm = GMMEstimator(label,args.c)
        gmms.append(gmm)
        gmm.fit(data)
        
    if args.test:
        testfile = args.test.readlines()
        testfile = sepTest(testfile)
        gmmpredictions = []
        for gmm in gmms:
            gmmpredictions.append(gmm.predict(testfile))
        
        gmmmax  = maximum(gmmpredictions[0],gmmpredictions[1])
        if args.o:
            with open(args.o,'w') as op:
                print "Writing out file %s"%(args.o)
                for prediction in gmmmax:
                    datap, label = prediction
                    op.write( str(datap[0]) + ' ' + str(datap[1]) +  ' ' + str(label) + '\n')
                print "Writeout done, finished!"

def maximum(l1,l2):
    if len(l1) != len(l2):
        return 
    ret= []
    for i in range(len(l1)):
        maxi1,datap1,gmmlabel1 = l1[i]
        maxi2,datap2,gmmlabel2 = l2[i]
        if maxi1 > maxi2:
            ret.append((datap1,gmmlabel1))
        else:
            ret.append((datap2,gmmlabel2))
    return ret
    

def getpredictions(l1,l2,maximas):
    if len(l1) != len(l2):
        return
    ret = []
    for i in range(len(maximas)):
        if maximas[i] == l1[i]:
            ret.append(object)
        
            
             
def parseargs():
    parser= argparse.ArgumentParser()
    parser.add_argument('--train',type=file,help='The train file')
    parser.add_argument('--test',type=file,help='The test file')
    parser.add_argument('--dev',type=file,help='The developement file')
    parser.add_argument('-c',type=int,help='The amount of components for each GMM',default=2)
    parser.add_argument('-o',type=str,help='Output file which will be produced with the labelled test data')    
    args=parser.parse_args()
    return args

def separateTrain(f):
    trainclasses = {}
    for itm  in f:
        a,b,c = itm.split()
#         Just want to use the labels as indices for the arrays so they need to go from 0 to n
        if c in trainclasses:
            oldl = trainclasses[c]
            oldl.append(np.array([float(a),float(b)]))
        else:
            trainclasses[c] = [np.array([float(a),float(b)])]
    return trainclasses

        
def sepTest(f):
    containers = []
    for itm  in f:
        a,b = itm.split()
        containers.append(np.array([float(a),float(b)]))
    return containers


if __name__ == '__main__':
    main()
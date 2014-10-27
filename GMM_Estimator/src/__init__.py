'''
Created on Sep 15, 2014

@author: richman
'''
from GMM_Estimator import GMMEstimator
import numpy as np
import argparse

def main():
    args = parseargs()
    if args.dev:
        devfile = args.dev.readlines()
        devfile = separateTrain(devfile)
    if args.train:
        trainfile=args.train.readlines()
        trainfile = separateTrain(trainfile)
    
    estimator = GMMEstimator([0,1])
    estimator.fit(trainfile)
    
    if args.test:
        testfile = args.test.readlines()
        testfile = sepTest(testfile)
        predictions = estimator.predict(testfile)
        if args.o:
            with open(args.o,'w') as op:
                for prediction in zip(testfile, predictions):
                    op.write(str(prediction[0].data) + ' ' +str(prediction[1])+'\n')
         
def parseargs():
    parser= argparse.ArgumentParser()
    parser.add_argument('--train',type=file,help='The train file')
    parser.add_argument('--test',type=file,help='The test file')
    parser.add_argument('--dev',type=file,help='The developement file')
    parser.add_argument('-o',type=str,help='Output file which will be produced with the labelled test data')    
    args=parser.parse_args()
    return args

class Container():
    
    def __init__(self,data,label):
        self.data = np.array(data)
        self.label = label
        
    def __len__(self):
        return len(self.data)

def separateTrain(f):
    containers = []
    for itm  in f:
        a,b,c = itm.split()
#         Just want to use the labels as indices for the arrays so they need to go from 0 to n
        containers.append(Container([float(a),float(b)],int(c)-1))
    return containers

        
def sepTest(f):
    containers = []
    for itm  in f:
        a,b = itm.split()
        containers.append(Container([float(a),float(b)],None))
    return containers


if __name__ == '__main__':
    main()
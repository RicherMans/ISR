'''
Created on Nov 5, 2014

@author: richman
'''
from argparse import ArgumentParser
import numpy as np
import decompositions

def main():
#     We sample 2 classes with 10 x,y coordinates each
    datapoints = np.random.sample((2,10,10))
#     A = np.array([[12, 1, 1], 
#         [2, 4, 1],
#         [3, -10, 10]])
    decompositions.lda(datapoints)
#     val = decompositions.eigdecomposition(datapoints)
    
def parseArgs():
    parser = ArgumentParser()
    parser.add_argument()
    return parser.parse_args()
if __name__ == "__main__":
    main()
     

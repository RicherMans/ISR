'''
Created on Nov 5, 2014

@author: richman
'''
from householder import *
import argparse.ArgumentParser

def main():
    A = np.array([[12, 1, 1], 
        [-2, 4, 1],
        [3, 3, -1]])
    val = eigdecomposition(np.array(A))
    
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args()
if __name__ == "__main__":
    main()
     

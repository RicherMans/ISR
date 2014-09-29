import argparse
import numpy as np
from fractions import Fraction

def main():
    args = parseArgs()
    print "\nInput Matrix : \n%s\n"%(args.mat)
    l, u, p = scaledPivot(args.mat)
    print "\nL Matrix : \n%s\n" % (l)
    print "\nU Matrix : \n%s\n" % (u)
#     print "\nP Matrix : \n%s\n" % (p)
    
    

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('mat', type=mattype)
    args = parser.parse_args()
    return args
    
    
    
def mattype(s):
    try:
        outersplits = s.split(';')
#         Need to have a square matrix
        singlelen = len(outersplits)
        maps = []
        for outsplit in outersplits:
            if singlelen != len(outsplit.split(',')):
                raise argparse.ArgumentTypeError()
            maps.append(map(float, outsplit.split(',')))
        return np.array(maps, dtype=Fraction)
    except:
        raise argparse.ArgumentTypeError("Input type must be something like: a,b,c;d,e,f;g,h,j format. Needs to be a square matrix")

def scaledPivot(mat):
#     p is the permutation matrix which keeps track of any row level interchange
    p = []
#     row max is the maximum element in each row
    rowmax = []
    
    for rowind in range(len(mat)):
        p.append(rowind)
        rowmax.append(max(abs(mat[rowind])))
    for k in range(len(mat) - 1):
        colmat = mat[:, k]
        maxcolval = -1
        maxcolind = -1
        for i in range(len(colmat)):
            tmpval = abs(colmat[p[i]] / rowmax[p[i]])
            if tmpval > maxcolval:
                maxcolval = tmpval
                maxcolind = i
#         Swap the current iteration index with the maxval
        p[k], p[maxcolind] = p[maxcolind], p[k]
#         I use p to iterate because otherwise a swap would be necessary for the p[i] p[j] rows
        for i in range(k + 1, len(p)):
            z = Fraction(colmat[p[i]] / colmat[p[k]]).limit_denominator()
            mat[p[i]][k] = z
            for j in range(k + 1, len(p)):
                mat[p[i]][j] = Fraction(mat[p[i]][j] - z * mat[p[k]][j]).limit_denominator()
        print "Iteration %i : maxcolvalue ( After division ) : %.2f \n%s \n" % (k + 1, maxcolval , mat)
    pmat = np.zeros(shape=mat.shape, dtype=Fraction)
    for i in range(len(pmat[0])):
        pmat[i][p[i]] = 1
#         For the result we need to rotate our matrix by using the permutation matrix p
    p_dot_mat = np.dot(pmat, mat)
#     Somehow an conversion by using the dtype parameter doesnt work, so need to convert explicitly
# Note that the value for 0 will be 0/1, which is still correct altough bad to read
    fract_p_dot_mat = np.array([[Fraction(p_dot_mat[j][i]).limit_denominator() for i in range(len(p_dot_mat[0]))] for j in range(len(p_dot_mat))])
    l = fract_p_dot_mat * np.tril(np.ones(fract_p_dot_mat.shape, dtype=Fraction),-1)+np.eye(fract_p_dot_mat.shape[0],dtype=Fraction)
    u = fract_p_dot_mat * np.triu(np.ones(fract_p_dot_mat.shape, dtype=Fraction),0)
    print np.dot(l,u)
    
    return (l, u, pmat)

if __name__ == '__main__':
    main()

"""File            cheney-householder.py
Author        Ernesto P. Adorio, Ph.D.
                  University of the Philippines Extenstion Program at Clark Field
                  Pampanga, the Philippines
Revisions    2009.01.20 first version
References
        Kincaid and Cheney, "Numerical Analysis",  2nd Ed. pp. 299-303
        Brooks Coole Publishing
"""
 
from math import sqrt
import numpy as np
 
def cheneyhouseholder(mat):
    # Given mat, determines a factorization Q mat = R or mat = Q'R.
    # Remove the verbose print lines if used in a library.
    nrows, ncols = mat.shape
    Q = np.zeros(shape=(nrows, nrows))
    R = np.zeros(shape=(nrows, ncols))
    for j in range(nrows - 1):
        # compute the column E^2 norm
        x = mat[:,j] # jth column vector.
        beta = -sqrt(sum(x * x for x in x))
        print "x = ", x
        print "beta=", beta
 
        
#         Ui = imvv
#         Ui = matprependidentity(Ui, j)
#         print "resolving Ui:"
#         print "U_", j , "= ",
#         for u in Ui: print u 
#   
#         UiA = matprod(Ui, mat)
#         print "U_", j, "mat:"
#         matprint(UiA)
#         mat = UiA
#         if j == 0:
#             Q = Ui
#         else:
#             Q = matmul(Ui, Q)
#         print "Q:"
#         matprint(Q)
#         R = UiA
    return Q, R
 
 
if __name__ == "__main__":
     A = [[12, -51, 4],  # Wikipedia
         [6, 167, -68],
         [-4, 24, -41]]
     A = [[63, 41, -88],  # Cheney
         [42, 60, 51],
        [0, -28, 56],
        [126, 82, -71]]
     print "A : %s"%(A) 
     Q, R = cheneyhouseholder(np.array(A))
     print "R:"
#      matprint(R)
     print("QR:")
#      matprint(matmul(Q, A))

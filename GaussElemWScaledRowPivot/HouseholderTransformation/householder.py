from math import sqrt
import numpy as np
 
def gram_schmidt(mat):
    # Given mat, determines a factorization q mat = r or mat = q'r.
    # Remove the verbose print lines if used in a library.
    nrows, ncols = mat.shape
    a = np.copy(mat)
    q = np.zeros(shape=(nrows, nrows),dtype=np.float)
    r = np.zeros(shape=(nrows, ncols),dtype=np.float)
    for k in range(nrows):
        # compute the column E^2 norm
        x = a[:, k]  # jth column vector.
        r[k,k] = sqrt(sum(x * x for x in x))
        for j in range(ncols):
            q[j,k]= a[j,k]/r[k,k]
        for i in range(k+1,nrows):
            s=0
            for j in range(ncols):
                s+= a[j,i] * q[j,k]
            r[k,i] = s
            for j in range(ncols):
                a[j,i] = a[j,i]-r[k,i]*q[j,k]
    return q, r


def eigdecomposition(mat):
    row,col = mat.shape
    eigenvalues=np.zeros(row)
#     print np.linalg.eigvals(mat)
    for m in reversed(range(1,row)):
        for _ in range(10):
#             q,r = np.linalg.qr(mat - mat[m,m]*np.eye(row) )
            q,r = gram_schmidt(mat - mat[m,m]*np.eye(row))
            mat = np.dot(r,q)+mat[m,m]*np.eye(row)
        eigenvalues[m] = mat[m,m]
    eigenvalues[0]= mat[0,0]
    print eigenvalues


from math import sqrt

import numpy as np


def poweig(A, x0, maxiter=100, ztol=1.0e-5, mode=0, teststeps=1):
    """
    Performs iterative power method for dominant eigenvalue.
     A  - input matrix.
     x0 - initial estimate vector.
     maxiter - maximum iterations
     ztol - zero comparison.
     mode:
       0 - divide by last nonzero element.
       1 - unitize.
    Return value:
     eigenvalue, eigenvector
    """
    m = len(A)
    xi = x0[:] 
 
    for n in range(maxiter):
        # matrix vector multiplication.
        xim1 = xi[:]
        for i in range(m):
            xi[i] = 0.0
            for j in range(m):
                xi[i] += A[i][j] * xim1[j]
        if mode == 0:
            vlen = sqrt(sum([xi[k] ** 2 for k in range(m)]))
            xi = [xi[k] / vlen for k in range(m)]
        elif mode == 1:
            for k in range(m - 1, -1, -1):
                c = abs(xi[k])
                if c > 1.0e-5:
                    xi = [xi[k] / c for k in range(m)]
                    break
        # early termination test.
        if n % teststeps == 0:
            S = sum([xi[k] - xim1[k] for k in range(m)])
            if abs(S) < ztol:
                break
    # Compute Rayleigh quotient.
    numer = sum([xi[k] * xim1[k] for k in range(m)])
    denom = sum([xim1[k] ** 2 for k in range(m)])
    xlambda = numer / denom
    return xlambda, xi
 

def lu_decomposition(A):
    """Performs an LU Decomposition of A (which must be square)                                                                                                                                                                                        
    into PA = LU. The function returns P, L and U."""
    
    def pivot_matrix(M):
        """Returns the pivoting matrix for M, used in Doolittle's method."""
        m = len(M)
    
        # Create an identity matrix, with floating point values                                                                                                                                                                                            
        id_mat = [[float(i == j) for i in xrange(m)] for j in xrange(m)]
    
        # Rearrange the identity matrix such that the largest element of                                                                                                                                                                                   
        # each column of M is placed on the diagonal of of M                                                                                                                                                                                               
        for j in xrange(m):
            row = max(xrange(j, m), key=lambda i: abs(M[i][j]))
            if j != row:
                # Swap the rows                                                                                                                                                                                                                            
                id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
    
        return id_mat
    
    def mult_matrix(M, N):
        """Multiply square matrices of same dimension M and N"""
    
        # Converts N into a list of tuples of columns                                                                                                                                                                                                      
        tuple_N = zip(*N)
    
        # Nested list comprehension to calculate matrix multiplication                                                                                                                                                                                     
        return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]
    n = len(A)

    # Create zero matrices for L and U                                                                                                                                                                                                                 
    L = [[0.0] * n for i in xrange(n)]
    U = [[0.0] * n for i in xrange(n)]

    # Create the pivot matrix P and the multipled matrix PA                                                                                                                                                                                            
    P = pivot_matrix(A)
    PA = mult_matrix(P, A)

    # Perform the LU Decomposition                                                                                                                                                                                                                     
    for j in xrange(n):
        # All diagonal entries of L are set to unity                                                                                                                                                                                                   
        L[j][j] = 1.0

        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}                                                                                                                                                                                      
        for i in xrange(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in xrange(i))
            U[i][j] = PA[i][j] - s1

        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )                                                                                                                                                                  
        for i in xrange(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in xrange(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return (P, L, U)
     

def gram_schmidt_qr(mat):
    # Given mat, determines a factorization q mat = r or mat = q'r.
    # Remove the verbose print lines if used in a library.
    nrows, ncols = mat.shape
    a = np.copy(mat)
    q = np.zeros(shape=(nrows, nrows), dtype=np.float)
    r = np.zeros(shape=(nrows, ncols), dtype=np.float)
    for k in range(nrows):
        # compute the column E^2 norm
        x = a[:, k]  # jth column vector.
        r[k, k] = sqrt(sum(x * x for x in x))
        for j in range(ncols):
            q[j, k] = a[j, k] / r[k, k]
        for i in range(k + 1, nrows):
            s = 0
            for j in range(ncols):
                s += a[j, i] * q[j, k]
            r[k, i] = s
            for j in range(ncols):
                a[j, i] = a[j, i] - r[k, i] * q[j, k]
    return q, r


def lda(datapoints,threshold = 0.01):
    '''
    datapoints is the given data, with shape [class featuredim1 featuredim2]
    threshold is the percentual treshold for removing the top eigenvalues, usually 0.01, which means that eigenvalues less than 1% 
    of the overall eigenvalue sum will be removed
    '''
    cl,m,n = datapoints.shape
    mean_vectors=np.zeros((cl,m))
    for i in range(cl):
        mean_vectors[i] = np.mean(datapoints[i], axis=0)
#         print('Mean Vector class %s: %s\n' % (i,mean_vectors[i]))
#     Calcuate Within Scatter matrix:
    s_w = np.zeros((m,n))
    for i in range(cl):
        curmat = datapoints[i]
        for row in curmat:
#             Get the current col
            col = row.reshape(n,1)
            meancol = mean_vectors[i].reshape(n,1)
            s_w += (col-meancol).dot((col-meancol).T)
#         print ('Within class covaricance Scatter : ',s_w[i])
    overall_mean = np.mean(mean_vectors, axis=0)
    s_b = np.zeros((m,n))
    for i in range(cl):
        samp_size = len(datapoints[i])
        mean_row = mean_vectors[i].reshape(n,1)
        s_b += samp_size*(mean_row - overall_mean).dot((mean_row-overall_mean).T)
    
    eig_vals,eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs,key=lambda k: k[0] ,reverse=True)
    eigv_sum = sum(eig_vals)
    maxindexTopVals = -1
    for i,j in enumerate(eig_pairs):
        if (j[0]/eigv_sum).real<threshold:
            maxindexTopVals = i-1 
            break
    w= np.hstack(eig_pairs[i][1].reshape(n,1) for i in range(maxindexTopVals))
    
    print w.T.shape
    print datapoints.shape
    print datapoints.reshape(cl*n,m).shape
    x_lda = w.T.dot(datapoints.reshape(1,cl*n,m)).T
    print x_lda.shape
def eigdecomposition(mat):
    row, col = mat.shape
    eigenvalues = np.zeros(row)
#     print np.linalg.eigvals(mat)
    for m in reversed(range(1, row)):
        for _ in range(10):
#             q,r = np.linalg.qr(mat - mat[m,m]*np.eye(row) )
            q, r = gram_schmidt_qr(mat - mat[m, m] * np.eye(row))
            mat = np.dot(r, q) + mat[m, m] * np.eye(row)
        eigenvalues[m] = mat[m, m]
    eigenvalues[0] = mat[0, 0]
    return eigenvalues

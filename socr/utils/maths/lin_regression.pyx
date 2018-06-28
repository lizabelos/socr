# Code translated from Transkribus BaseLine Evaluation Scheme
# Original Author : Tobias Gruening tobias.gruening.hro@gmail.com

import numpy as np
import math

cpdef double[:] calcLine(int[:] xPoints, int[:] yPoints):
    cdef int dimA = xPoints.shape[0]
    cdef double minX = 10000
    cdef double maxX = 0
    cdef double sumX = 0.0

    cdef double[:,:] A = np.zeros((dimA, 2))
    cdef double[:] Y = np.zeros(dimA)

    for i in range(0, dimA):
        A[i][0] = 1.0
        A[i][1] = xPoints[i]
        minX = min(minX, xPoints[i])
        maxX = max(maxX, xPoints[i])
        sumX += xPoints[i]
        Y[i] = yPoints[i]

    if maxX - minX < 2:
        result = np.zeros((2))
        result[0] = sumX / dimA
        result[1] = math.inf
        return result

    return solveLin(A, Y)


cpdef double[:] solveLin(double[:,:] mat1, double[:] Y):
    cdef double[:,:] mat1T = transpose(mat1)
    cdef double[:,:] multLS = multiply2D(mat1T, mat1)
    cdef double[:] multRS = multiply1D(mat1T, Y)

    if multLS.shape[0] != 2:
        raise Exception("LinRegression Error: Matrix not 2x2")

    cdef double[:,:] inv = np.zeros((2,2))
    cdef double n = (multLS[0][0] * multLS[1][1] - multLS[0][1] * multLS[1][0])
    if n < 1E-9:
        print("LinRegression Error: Numerically unstable.")
        result = np.zeros((2))
        result[0] = mat1[0][1]
        result[1] = math.inf
        return result

    cdef double fac = 1.0 / n
    inv[0][0] = fac * multLS[1][1]
    inv[1][1] = fac * multLS[0][0]
    inv[1][0] = -fac * multLS[1][0]
    inv[0][1] = -fac * multLS[0][1]

    cdef double[:] res = multiply1D(inv, multRS)
    return res


cdef double[:,:] transpose(double[:,:] A):
    cdef double[:,:] res = np.zeros((A.shape[0], A.shape[1]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            res[j][i] = A[i][j]

    return res


cdef double[:] multiply1D(double[:,:] A, double[:] x):
    if A.shape[1] != x.shape[0]:
        raise Exception("LinRegression Error: Matrix dimension mismatch.")

    cdef double[:] res = np.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            res[i] += x[j] * A[i][j]

    return res


cdef double[:,:] multiply2D(double[:,:] A, double[:,:] B):
    if A.shape[1] != B.shape[0]:
        raise Exception("LinRegression Error: Matrix dimension mismatch.")

    cdef double[:,:] res = np.zeros((A.shape[0],B.shape[1]))
    for i in range(0, A.shape[0]):
        for j in range(0, B.shape[1]):
            for k in range(0, B.shape[0]):
                res[i][j] += B[k][j] * A[i][k]

    return res
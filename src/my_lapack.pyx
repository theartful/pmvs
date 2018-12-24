cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *
from libc.stdlib cimport malloc, free
# http://www.math.utah.edu/software/lapack/


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1, :] matmatmul(double[::1, :] amat, double[::1, :] bmat):
    cdef:
        int m = amat.shape[0]
        int lda = amat.shape[0]
        int ldc = amat.shape[0]
        int k = amat.shape[1]
        int ldb = amat.shape[1]
        int n = bmat.shape[1]
        double[::1, :] cmat = np.empty((m, n), float, order='F')
        double alpha = 1.0
        double beta = 0.0

    # http://www.math.utah.edu/software/lapack/lapack-blas/dgemm.html
    dgemm('N', 'N', &m, &n, &k, &alpha, &amat[0, 0], &lda,
          &bmat[0, 0], &ldb, &beta, &cmat[0, 0], &ldc)

    return cmat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] matvmul(double[::1, :] amat, double[:] bvec):

    cdef:
        int m = amat.shape[0]
        int n = amat.shape[1]
        int inc = 1
        double[:] cmat = np.empty(m, float, order='F')
        double alpha = 1.0
        double beta = 0.0

    # http://www.math.utah.edu/software/lapack/lapack-blas/dgemv.html
    dgemv('N', &m, &n, &alpha, &amat[0, 0], &m, &bvec[0], &inc,
          &beta, &cmat[0], &inc)
    return cmat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double vdot(double[:] avec, double[:] bvec):
    cdef:
        double out = 0.0
        int n = avec.shape[0]
        int inc = 1

    # http://www.mathkeisan.com/usersguide/man/ddot.html
    out = ddot(&n, &avec[0], &inc, &bvec[0], &inc)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] cdot(double[:] vec, double c):
    cdef int i
    cdef double[:] vec2 = vec.copy_fortran()
    for i in range(vec2.shape[0]):
        vec2[i] *= c
    return vec2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] cdot_nocp(double[:] vec, double c):
    cdef int i
    for i in range(vec.shape[0]):
        vec[i] *= c
    return vec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] solve(double[::1, :] amat, double[:] bvec):
    cdef:
        int n = amat.shape[0]
        int[:] ipiv = np.empty(n, np.int32, order='F')
        int info
        int nrhs = 1

    dgesv(&n, &nrhs, &amat[0, 0], &n, &ipiv[0], &bvec[0], &n, &info)
    return bvec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1,:] solve2(double[::1, :] amat, double[::1,:] bvec):
    cdef:
        int n = amat.shape[0]
        int[:] ipiv = np.empty(n, np.int32, order='F')
        int info
        int nrhs = bvec.shape[1]

    dgesv(&n, &nrhs, &amat[0, 0], &n, &ipiv[0], &bvec[0, 0], &n, &info)
    return bvec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double norm(double[:] avec):
    cdef:
        double out = 0.0
        int n = avec.shape[0]
        int inc = 1

    # http://www.mathkeisan.com/usersguide/man/dnrm2.html
    out = dnrm2(&n, &avec[0], &inc)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1, :] matadd(double[::1, :] amat, double[::1, :] bmat):
    cdef double[::1,:] cmat = amat.copy_fortran()
    cdef int x, y
    for x in range(bmat.shape[0]):
        for y in range(bmat.shape[1]):
            cmat[x, y] += bmat[x, y]
    return cmat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1, :] matsub(double[::1, :] amat, double[::1, :] bmat):
    cdef double[::1,:] cmat = amat.copy_fortran()
    cdef int x, y
    for x in range(bmat.shape[0]):
        for y in range(bmat.shape[1]):
            cmat[x, y] -= amat[x, y]
    return cmat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] vadd(double[:] v1, double[:] v2):
    cdef double[:] v3 = v1.copy_fortran()
    cdef int x
    for x in range(v2.shape[0]):
        v3[x] += v2[x]
    return v3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] vsub(double[:] v1, double[:] v2):
    cdef double[:] v3 = v1.copy_fortran()
    cdef int x
    for x in range(v2.shape[0]):
        v3[x] -= v2[x]
    return v3

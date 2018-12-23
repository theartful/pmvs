cpdef double[::1, :] matmatmul(double[::1, :] amat, double[::1, :] bmat)
cpdef double[:] matvmul(double[::1, :] amat, double[:] bvec)

cpdef double[:] solve(double[::1, :] amat, double[:] bvec)
cpdef double[::1,:] solve2(double[::1, :] amat, double[::1,:] bvec)

cpdef double norm(double[:] avec)

cpdef double[::1, :] matadd(double[::1, :] amat, double[::1, :] bmat)
cpdef double[::1, :] matsub(double[::1, :] amat, double[::1, :] bmat)

cpdef double[:] vadd(double[:] v1, double[:] v2)
cpdef double[:] vsub(double[:] v1, double[:] v2)

cpdef double vdot(double[:] avec, double[:] bvec)
cpdef double[:] cdot(double[:] vec, double c)
cpdef double[:] cdot_nocp(double[:] vec, double c)
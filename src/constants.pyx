from libc.math cimport cos
from libc.math cimport pi


cdef public int PATCH_GRID_SIZE = 5
# minimum number of T(p) for a patch to be accepted
cdef public int T_THRESHOLD = 3
cdef public int CELL_SIZE = 2
cdef public int FEATURE_GRID_SIZE = 32
cdef public int FEATURE_PER_GRID = 4
cdef public int RESIZE_FACTOR = 1
# maximum distance between a feature and the epipolar line
cdef public int CLEARANCE = 2

cdef public float COS_MIN_ANGLE = cos(10 * pi / 180)
cdef public float COS_MAX_ANGLE = cos(60 * pi / 180)

cdef public float THRESHOLD1 = 0.6
cdef public float THRESHOLD2 = 0.7

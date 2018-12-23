cdef public int PATCH_GRID_SIZE
# minimum number of T(p) for a patch to be accepted
cdef public int T_THRESHOLD
cdef public int CELL_SIZE
cdef public int FEATURE_GRID_SIZE
cdef public int FEATURE_PER_GRID
cdef public int RESIZE_FACTOR
# maximum distance between a feature and the epipolar line
cdef public int CLEARANCE

cdef public float COS_MIN_ANGLE
cdef public float COS_MAX_ANGLE

cdef public float THRESHOLD1
cdef public float THRESHOLD2

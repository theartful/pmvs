from data_structs cimport *
from constants cimport *
import numpy as np
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.feature import corner_harris


cpdef detect_features(Image img):
    cdef double[:,:] dog_img = _dog(img)
    cdef double[:,:] harris_img = _harris(img)

    height, width = img.data.shape[0], img.data.shape[1]
    cdef int x_cells = width // FEATURE_GRID_SIZE
    cdef int y_cells = height // FEATURE_GRID_SIZE

    cdef int j, i, init_y, init_x
    for j in range(x_cells):
        for i in range(y_cells):
            init_y = i * FEATURE_GRID_SIZE
            init_x = j * FEATURE_GRID_SIZE
            _fill_features(img, DOG, dog_img, init_y, init_x)
            _fill_features(img, HARRIS, harris_img, init_y, init_x)


cpdef double[:,:] _dog(Image img, double sig1=1, double sig2=np.sqrt(2)):
    gauss1 = gaussian(img.data, sigma=sig1, multichannel=True)
    gauss2 = gaussian(img.data, sigma=sig2, multichannel=True)
    tmp = np.abs(gauss1 - gauss2)
    tmp[img.silhouette == int(0)] = 0
    return rgb2gray(tmp)


cpdef double[:,:] _harris(Image img, double k=0.06):
    harris = corner_harris(rgb2gray(img.data), k=k)
    harris[img.silhouette == int(0)] = 0
    return harris


cpdef _fill_features(Image img, int feat_type, double[:,:] response_img, int init_y, int init_x):
    cdef double[:,:] response_cell = response_img[init_y: init_y + FEATURE_GRID_SIZE,
                                 init_x: init_x + FEATURE_GRID_SIZE]
    cdef long[:] highest_indices = np.argpartition(
                np.array(response_cell).reshape([-1, ]), -FEATURE_PER_GRID)[-FEATURE_PER_GRID:]

    cdef int index, y, x
    cdef Feature feature
    for index in highest_indices:
        y = index // FEATURE_GRID_SIZE
        x = index % FEATURE_GRID_SIZE
        if response_cell[y, x] == 0:
            continue
        feature = Feature(y + init_y, x + init_x, feat_type, img)
        img.add_feature(feature)

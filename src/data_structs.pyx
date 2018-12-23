import numpy as np
cimport numpy as np
from scipy.misc import imresize
from constants cimport *
from my_lapack cimport *
cimport cython


cpdef int DOG = 0
cpdef int HARRIS = 1


cdef class Camera:
    def __cinit__(self, np.ndarray[np.float64_t, ndim=2] camera_matrix):
        camera_matrix[2] *= RESIZE_FACTOR
        self.camera_matrix = camera_matrix.copy(order='F')
        self.pinv = np.array(np.linalg.pinv(self.camera_matrix), order='F')
        # compute optical center
        m_inverse = np.linalg.inv(self.camera_matrix[0:3, 0:3])
        col_4 = self.camera_matrix[:, 3].copy()
        self.optical_center = np.hstack((-np.dot(m_inverse, col_4), 1))
        tmp = camera_matrix[2].copy()
        tmp[-1] = 0
        tmp /= np.linalg.norm(tmp)
        self.optical_axis = tmp


cdef class Feature:
    def __cinit__(self, int image_y, int image_x, int feature_type, Image img):
        self.coord = np.array([image_x, image_y, 1], dtype=np.float)
        self.y = image_y
        self.x = image_x
        self.feature_type = feature_type
        self.img = img
        j = self.x // CELL_SIZE
        i = self.y // CELL_SIZE
        self.cell = self.img.cell(j=j, i=i)


cdef class Cell:
    def __cinit__(self, Image img, int x_init, int y_init):
        self.t_patches = []
        self.s_patches = []
        self.depth = 0
        self.x_initial = x_init
        self.y_initial = y_init
        self.img = img

    def as_np_array(self):
        return self.img[self.y_initial: self.y_initial + CELL_SIZE,
               self.x_initial: self.x_initial + CELL_SIZE]


cdef class Patch:
    def __cinit__(self, Image r_image, Cell cell=None, double[:] normal=None, double[:] center=None):
        self.t_images = []
        self.s_images = []
        self.r_image = r_image
        self.normal = normal
        self.center = center
        self.cell = cell

        tmp = np.dot(self.r_image.camera_matrix(), center)
        tmp = np.divide(tmp, tmp[-1])
        self.image_center = np.rint(tmp).astype(np.int32)
        
        cdef int step = (PATCH_GRID_SIZE - 1) // 2
        cdef int[:] top_left = np.subtract(self.image_center, np.array([step, step, 0])).astype(np.int32)
        self.source_cell = np.array(
            [[r_image.get_pixel(y=top_left[1] + i, x=top_left[0] + j)
              for i in range(PATCH_GRID_SIZE)] for j in range(PATCH_GRID_SIZE)], dtype=np.int32
        )


cdef class Image:
    def __cinit__(self, data, silhouette, camera):
        self.data = imresize(data, 1.0 / RESIZE_FACTOR).astype(np.int32)
        self.silhouette = \
            imresize(silhouette.astype(np.int), 1.0 / RESIZE_FACTOR).astype(np.int32)
        self.camera = camera
        [height, width, _] = data.shape
        self.x_cells = width // CELL_SIZE
        self.y_cells = height // CELL_SIZE
        self.cells = [[None] * self.x_cells for _ in range(self.y_cells)]
        self.dog_features = []
        self.harris_features = []
        self._zeros = np.array([0, 0, 0], dtype=np.int32)

    cdef Cell cell(self, int i, int j):
        if self.cells[i][j] is None:
            self.cells[i][j] = Cell(
                x_init=j * CELL_SIZE,
                y_init=i * CELL_SIZE,
                img=self)
        return self.cells[i][j]

    cdef void add_feature(self, Feature feat):
        feat.img = self
        if feat.feature_type == DOG:
            self.dog_features.append(feat)
        elif feat.feature_type == HARRIS:
            self.harris_features.append(feat)

    cdef double[::1,:] camera_matrix(self):
        return self.camera.camera_matrix.copy_fortran()

    cdef double[::1,:] pinv(self):
        return self.camera.pinv.copy_fortran()

    cdef double[:] optical_center(self):
        return self.camera.optical_center.copy_fortran()

    cdef double[:] optical_axis(self):
        return self.camera.optical_axis.copy_fortran()

    cdef bint check_in_image(self, double y, double x):
        if y < 0 or x < 0:
            return False
        if y + 0.5 >= self.data.shape[0] or x + 0.5 >= self.data.shape[1]:
            return False
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True) 
    cdef int[:] get_pixel(self, double y, double x):
        return self.nearest_neighbor(y, x)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True) 
    cdef int[:] nearest_neighbor(self, double y, double x):
        if y < 0 or x < 0:
            return self._zeros
        if y + 0.5 >= self.data.shape[0] or x + 0.5 >= self.data.shape[1]:
            return self._zeros
        return self.data[int(round(y)), int(round(x))]


cdef class ImagesManager:
    def __cinit__(self, images, silhouettes, matrices):
        self.images = [Image(images[i], silhouettes[i], Camera(matrices[i]))
                       for i in range(len(images))]
        self._fundamental_matrices = {}
        self._camera_distances = {}
        self.patches = []

    cdef double[::1,:] fundamental_matrix(self, Image img1, Image img2):
        """
        returns the fundamental matrix F relating image(j) to image(i)
        such that if x \in image(i) and x' a corresponding point \in image(j)
        then <x', Fx> = 0. in other words, Fx is the epipolar line in image(j)
        corresponding to x
        :param img1: the first image
        :param img2: the second image
        :return: fundamental matrix F
        """
        if (img1, img2) in self._fundamental_matrices:
            return self._fundamental_matrices[(img1, img2)]

        cdef double[::1,:] p_pdash = matmatmul(img2.camera_matrix(), img1.camera.pinv)
        cdef double[::1,:] fun_mat = matmatmul(_skew_form(epipole(img1, img2)), p_pdash)
        self._fundamental_matrices[(img1, img2)] = fun_mat
        self._fundamental_matrices[(img2, img1)] = np.array(fun_mat.T, order='F')
        return fun_mat

    cdef double camera_distance(self, Camera cam1, Camera cam2):
        if cam1 == cam2:
            return 0
        if (cam1, cam2) in self._camera_distances:
            return self._camera_distances[(cam1, cam2)]
        cdef double dis = norm(vsub(cam1.optical_center, cam2.optical_center))
        cdef double cos_angle = np.dot(cam1.optical_axis[0:3], cam2.optical_axis[0:3])
        cdef double tot_dis = float('inf') if cos_angle == 0 else dis / cos_angle
        self._camera_distances[(cam1, cam2)] = tot_dis
        self._camera_distances[(cam2, cam1)] = tot_dis
        return tot_dis


cdef double[::1,:] _skew_form(double[:] e):
    """
    calculates [e]x such that: e cross_product v =  [e]x matmul v
    :param e: 3D vector
    :return: skew matrix form for the cross product
    """
    return np.array([[0, -e[2], e[1]],
                    [e[2], 0, -e[0]],
                    [-e[1], e[0], 0]], order='F')

cdef double[:] epipole(Image img1, Image img2):
    """
    returns the projection of the optical center of image(i) in image(j)
    :param img1: the first image
    :param img2: the second image
    :return: the projection of the optical center of image(i) in image(j)
    """
    return np.dot(img2.camera_matrix(), img1.optical_center())

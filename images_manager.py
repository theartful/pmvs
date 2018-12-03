import numpy as np


class Camera:
    def __init__(self, camera_matrix):
        assert len(camera_matrix.shape) == 2
        assert camera_matrix.shape[0] == 3
        assert camera_matrix.shape[1] == 4

        self.camera_matrix = camera_matrix
        # compute optical center
        m_inverse = np.linalg.inv(camera_matrix[0:3, 0:3])
        col_4 = camera_matrix[:, 3]
        self.optical_center = np.hstack((-m_inverse.dot(col_4), 1))
        self.pinv = np.linalg.pinv(self.camera_matrix)


class Feature:
    def __init__(self, image_y, image_x, feature_type, img=None):
        self.y = image_y
        self.x = image_x
        self.feature_type = feature_type
        self.img = img

    def coord(self):
        return np.array([self.x, self.y, 1])


class Cell:
    def __init__(self,x_init,y_init):
        self.t_patches = []
        self.s_patches = []
        self.depth = 0
        self.x_intial = x_init
        self.y_intial = y_init



class Patch:
    def __init__(self, r_image, cell=None, normal=None, center=None):
        self.t_images = []
        self.s_images = []
        self.r_image = r_image
        self.normal = normal
        self.center = center
        self.cell = cell


class Image:
    def __init__(self, data, silhouette=None, camera=None, cell_size=2):
        self.data = data
        self.silhouette = silhouette
        self.camera = camera
        self.cell_size = cell_size
        [height, width, _] = data.shape
        self.x_cells = width // cell_size
        self.y_cells = height // cell_size
        self.cells = [[None] * self.x_cells for _ in range(self.y_cells)]
        self.dog_features = []
        self.harris_features = []

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def cell(self, i, j):
        if self.cells[i][j] is None:
            self.cells[i][j] = Cell(x_init =j*self.cell_size,y_init = i * self.cell_size)
        return self.cells[i][j]

    def add_feature(self, feat):
        feat.img = self
        if feat.feature_type == 'dog':
            self.dog_features.append(feat)
        elif feat.feature_type == 'harris':
            self.harris_features.append(feat)

    def camera_matrix(self):
        if self.camera is None:
            return None
        return self.camera.camera_matrix

    def optical_center(self):
        if self.camera is None:
            return None
        return self.camera.optical_center

    def interpolate(self, y, x):
        """
        perform bilinear interpolation to find the color at (y, x)
        """
        x1 = np.floor(x).astype(np.int)
        x2 = x1 + 1
        y1 = np.floor(y).astype(np.int)
        y2 = y1 + 1

        factor = 1.0 / ((x2 - x1) * (y2 - y1))
        v1 = np.array([x2 - x, x - x1])
        v2 = np.array([self.data[y1, x1] * (y2 - y) +
                       self.data[y2, x1] * (y - y1),
                       self.data[y1, x2] * (y2 - y) +
                       self.data[y2, x2] * (y - y1)])
        return factor * v1.dot(v2)


class ImagesManager:
    def __init__(self, images, silhouettes, matrices):
        self.images = [Image(images[i], silhouettes[i], Camera(matrices[i]))
                       for i in range(len(images))]
        self._fundamental_matrices = {}
        self.patches = []

    def image(self, index):
        return self.images[index]

    def fundamental_matrix(self, img1, img2):
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

        p_pdash = img2.camera_matrix().dot(img1.camera.pinv)
        fun_mat = self._skew_form(self.epipole(img1, img2)).dot(p_pdash)
        self._fundamental_matrices[(img1, img2)] = fun_mat
        self._fundamental_matrices[(img2, img1)] = fun_mat.T
        return fun_mat

    @staticmethod
    def _skew_form(e):
        """
        calculates [e]x such that: e cross_product v =  [e]x dot v
        :param x: 3D vector
        :return: skew matrix form for the cross product
        """
        return np.array([[0, -e[2], e[1]],
                         [e[2], 0, -e[0]],
                         [-e[1], e[0], 0]])

    def __getitem__(self, item):
        return self.images[item]

    @staticmethod
    def epipole(img1, img2):
        """
        returns the projection of the optical center of image(i) in image(j)
        :param img1: the first image
        :param img2: the second image
        :return: the projection of the optical center of image(i) in image(j)
        """
        return img2.camera_matrix().dot(img1.optical_center())

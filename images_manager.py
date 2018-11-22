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


class Feature:
    def __init__(self, image_y, image_x, feature_type):
        self.y = image_y
        self.x = image_x
        self.feature_type = feature_type


class Cell:
    def __init__(self, data):
        self.data = data
        self.features = []
        self.t_patches = []
        self.s_patches = []
        self.depth = 0

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value


class Image:
    def __init__(self, data, camera=None, cell_size=2):
        self.data = data
        self.camera = camera
        self.cell_size = cell_size
        [height, width, _] = data.shape
        x_cells = width // cell_size
        y_cells = height // cell_size
        self.cells = [[None] * x_cells] * y_cells

    def _get_cell(self, i, j):
        """
        private method to create cell indexed(i, j)
        :param i: index of the cell in the y axis
        :param j: index of the cell in the x axis
        :return: image cell(i, j)
        """
        y = i * self.cell_size
        x = j * self.cell_size
        return Cell(self.data[y: y + self.cell_size, x: x + self.cell_size],
                    y, x)

    def cell(self, i, j):
        if self.cells[i][j] is None:
            self.cells[i][j] = self._get_cell(i, j)
        return self.cells[i][j]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

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
    def __init__(self, images, matrices):
        self.images = [Image(images[i], Camera(matrices[i]))
                       for i in range(len(images))]
        self._fundamental_matrices = {}

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

        camera_matrix_pinv = np.linalg.pinv(img1.camera_matrix())
        p_pdash = img2.camera_matrix().dot(camera_matrix_pinv)
        fun_mat = np.cross(self.epipole(img1, img2), p_pdash)
        self._fundamental_matrices[(img1, img2)] = fun_mat
        self._fundamental_matrices[(img2, img1)] = fun_mat.T
        return fun_mat

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


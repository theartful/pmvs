import images_manager
import numpy as np
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.feature import corner_harris


def detect_features(img, cell_size=32, features_per_cell=4):
    dog_img = rgb2gray(_dog(img.data) * (img.silhouette == 0))
    harris_img = corner_harris(rgb2gray(img.data), k=0.06) * \
                 (img.silhouette[:, :, 0] == 0)

    [height, width, _] = img.data.shape
    x_cells = width // cell_size
    y_cells = height // cell_size

    for j in range(x_cells):
        for i in range(y_cells):
            init_y = i * img.cell_size
            init_x = j * img.cell_size
            _fill_features(img.dog_features, dog_img, features_per_cell,
                           init_y, init_x, cell_size)
            _fill_features(img.harris_features, harris_img, features_per_cell,
                           init_y, init_x, cell_size)


def _dog(img, sig1=1, sig2=np.sqrt(2)):
    gauss1 = gaussian(img, sigma=sig1, multichannel=True)
    gauss2 = gaussian(img, sigma=sig2, multichannel=True)
    return np.abs(gauss1 - gauss2)


def _fill_features(features, response_img, k, init_y, init_x, cell_size):
    response_cell = \
        response_img[init_y: init_y + cell_size, init_x: init_x + cell_size]
    highest_indices = np.argpartition(response_cell.reshape([-1, ]), -k)[-k:]
    for index in highest_indices:
        y = index // cell_size
        x = index % cell_size
        if response_cell[y, x] == 0:
            break
        feature = images_manager.Feature(y + init_y, x + init_x)
        features.append(feature)

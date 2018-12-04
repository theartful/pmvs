import images_manager
import numpy as np
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.feature import corner_harris


def detect_features(img, cell_size=32, features_per_cell=4):
    dog_img = _dog(img)
    harris_img = _harris(img)

    [height, width, _] = img.data.shape
    x_cells = width // cell_size
    y_cells = height // cell_size

    for j in range(x_cells):
        for i in range(y_cells):
            init_y = i * cell_size
            init_x = j * cell_size
            _fill_features(img, 'dog', dog_img, features_per_cell,
                           init_y, init_x, cell_size)
            _fill_features(img, 'harris', harris_img, features_per_cell,
                           init_y, init_x, cell_size)


def _dog(img, sig1=1, sig2=np.sqrt(2)):
    gauss1 = gaussian(img.data, sigma=sig1, multichannel=True)
    gauss2 = gaussian(img.data, sigma=sig2, multichannel=True)
    tmp = np.abs(gauss1 - gauss2)
    tmp[img.silhouette == 0] = 0
    return rgb2gray(tmp)


def _harris(img, k=0.06):
    harris = corner_harris(rgb2gray(img.data), k=k)
    harris[img.silhouette == 0] = 0
    return harris


def _fill_features(img, feat_type, response_img, k, init_y, init_x, cell_size):
    response_cell = \
        response_img[init_y: init_y + cell_size, init_x: init_x + cell_size]
    highest_indices = np.argpartition(response_cell.reshape([-1, ]), -k)[-k:]
    for index in highest_indices:
        y = index // cell_size
        x = index % cell_size
        if response_cell[y, x] == 0:
            continue
        feature = images_manager.Feature(y + init_y, x + init_x, feat_type)
        img.add_feature(feature)

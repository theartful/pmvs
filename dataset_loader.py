import os.path
import numpy as np
from skimage.io import imread


def load(path):
    images = []
    silhouettes = []
    camera_matrices = []
    index = 0
    while os.path.isfile("{path}/images/{index:04}.ppm".format(path=path,
                                                               index=index)):
        img = imread("{path}/images/{index:04}.ppm".format
                     (path=path, index=index))
        silhouette = imread("{path}/silhouettes/{index:04}.pgm".format
                      (path=path, index=index)) == 0
        img[silhouette == 0] = 0
        images.append(img)
        silhouettes.append(silhouette)
        with open("{path}/calib/{index:04}.txt".format(path=path,
                                                       index=index)) as file:
            next(file)
            camera_matrix = np.zeros([3, 4])
            row = 0
            for line in file:
                array = [float(x) for x in line.split()]
                camera_matrix[row] = array
                row += 1
            camera_matrices.append(camera_matrix)
        index += 1
    return [images, silhouettes, camera_matrices]

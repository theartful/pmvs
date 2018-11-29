from images_manager import *
import numpy as np


def match_epipolar_consistency(feature, img_manager, d=2):
    consistent_features = []
    img = feature.img
    for element in img_manager:
        if element == img:
            continue

        if feature.feature_type == 'dog':
            features = element.dog_features
        else:
            features = element.harris_features

        fun_mat = img_manager.fundamental_matrix(img, element)
        epi_line = fun_mat.dot(np.array([feature.x, feature.y, 1]))
        con = d * np.sqrt(epi_line[0] ** 2 + epi_line[1] ** 2)

        for feat in features:
            if np.abs(np.array([feat.x, feat.y, 1]).dot(epi_line)) <= con:
                consistent_features.append(feat)

    return consistent_features


def get_patch_vectors(patch):
    """
    calculates the vectors (right, up) on the patch plane in world space
    so that moving along side one of them corresponds to moving a unit
    in the same direction in the reference image. in other words,
    the projections of (right, up) are (1, 0, 0) and (0, 1, 0) respectively
    :return: right, up
    """
    p = patch.r_image.camera_matrix()

    a = np.vstack([p, patch.normal])
    a_inverse = np.linalg.inv(a)

    right = a_inverse.dot(np.array([1, 0, 0, 0]))
    up = a_inverse.dot(np.array([0, 1, 0, 0]))

    d = p.dot(patch.center)[2]
    return right * d, up * d


def triangulate(feat1_coord, feat2_coord, proj1, proj2, fun_mat):
    line = fun_mat.dot(feat1_coord)
    line_perp = np.array([-line[0], line[1], line[0] * feat2_coord[0] -
                          line[1] * feat2_coord[1]])
    a = np.vstack([proj1, proj2.T.dot(line_perp)])
    b = np.array([feat1_coord[0], feat1_coord[1], feat1_coord[2], 0])
    return np.linalg.solve(a, b)




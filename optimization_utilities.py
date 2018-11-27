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


def update_patch_vectors(patch):
    p = patch.r_image.camera_matrix()

    m = np.vstack([p, patch.normal])
    m_inverse = np.linalg.inv(m)

    right = m_inverse.dot(np.array([1, 0, 0, 0]))
    up = m_inverse.dot(np.array([0, 1, 0, 0]))

    d = p.dot(patch.center)[2]
    return right * d, up * d



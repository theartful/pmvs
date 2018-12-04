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
    line_perp = np.array([-line[1], line[0], (line[1] * feat2_coord[0] -
                          line[0] * feat2_coord[1]) / feat2_coord[2]])
    a = np.vstack([proj1, proj2.T.dot(line_perp)])
    b = np.array([feat1_coord[0], feat1_coord[1], feat1_coord[2], 0])
    return np.linalg.solve(a, b)

def correlation_coefficient(cell1, cell2):
    """
    calculates normlized cross correlation for two cells
    @params: source and destination cells
    @return: normlized cross correlation of (cell1,cell2)
    """
    product = np.mean((cell1 - cell1.mean()) * (cell2 - cell2.mean()))
    stds = cell1.std() * cell2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def ncc(img,p,right,up):
    """
    takes source patch, traverse around cell center and project
    each point on the destination image
    accumaltes pixel values in auxiliary cell
    and computes the ncc between the aux cell and the cell of the source patch
    """
    c = p.center
    cell_size = img.cell_size
    camera_matrix = img.camera_matrix()
    step = (cell_size -1)/2
    top_left =  c - (right  + up ) * step
    source_cell = img.data[p.cell.y_intial:p.cell.y_intial+cell_size,p.cell.x_intial:p.cell.x_intial+cell_size]
    p_cell = np.zeros((cell_size,cell_size,3))
    for y in range(cell_size):
        for x in range(cell_size):
            current_pos = top_left + y * up + x * right
            projected_cpos =  camera_matrix.dot(current_pos)
            projected_cpos = projected_cpos / projected_cpos[-1]
            p_rgb = img.interpolate(x=projected_cpos[0],y=projected_cpos[1])
            p_cell[y][x] = p_rgb
    return correlation_coefficient(source_cell,p_cell)


def similarity_function(patch):
    """
    calculates accumlates ncc for source patch with t_images (truely visible images as said by essam)
    """
    right,up = get_patch_vectors(patch)
    accumlative_ncc = 0
    for img in patch.t_images:
        accumlative_ncc = accumlative_ncc + ncc(img,patch,right,up)
    accumlative_ncc = accumlative_ncc / len( patch.t_images)
    return accumlative_ncc

def optimize_similarity(params, *args):
    depth, alpha, beta = params
    patch = args[0]
    tmp = patch.center
    tmp /= tmp[-1]
    optical_center = patch.r_image.camera.optical_center
    optical_center /= optical_center[-1]
    unit_vector = (tmp - optical_center)
    unit_vector /= np.linalg.norm(unit_vector)
    center = optical_center + depth * unit_vector

    normal = np.array([np.sin(alpha) * np.cos(beta), \
                        np.sin(alpha) * np.sin(beta), \
                        np.cos(alpha), 0])
    patch.center = center
    patch.normal = normal
    return similarity_function(patch)

def set_patch_t_images(patch, images, alpha):
    right, up = get_patch_vectors(patch)
    for img in images:
        if(ncc(img,patch,right,up) >= alpha):
            c = patch.center
            camera_matrix = img.camera_matrix()
            c_projected = camera_matrix.dot(c)
            c_projected = c_projected / c_projected[-1]
            is_in = img.silhouette[int(c_projected[1]), int(c_projected[0]) ]
            print(is_in)
            if is_in:
                patch.t_images.append(img)


def set_t_images(patches,images,alpha):
    for p in patches:
        set_patch_t_images(p,images,alpha)

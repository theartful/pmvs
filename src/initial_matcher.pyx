from my_lapack cimport *
import time

import feature_detector
from dataset_loader import *
from common_functions cimport *
from data_structs cimport *
from constants cimport *
from libc.math cimport sqrt
from libc.math cimport abs
from cpython.exc cimport PyErr_CheckSignals


cdef void perform_matching(ImagesManager images_manager, bint detect_features=True):
    print("begin matching...")
    cdef list images = images_manager.images
    cdef Image img
    if detect_features:
        print("detecting features...")
        for img in images:
            feature_detector.detect_features(img)
        print("done detecting features!")

    print("begin feature matching...")
    cdef int i = 0
    cdef list relevant_images
    cdef list features
    cdef Feature feature
    cdef int patches_num
    for img in images:
        print("start img " + str(i))
        relevant_images = get_relevant_images(img, images_manager)
        features = img.dog_features + img.harris_features
        patches_num = len(images_manager.patches)
        for feature in features:
            construct_patch(images_manager, relevant_images, feature)
            PyErr_CheckSignals()
        i += 1
    print(str(len(images_manager.patches) - patches_num) +
              " patches created succesfully!")
    print("current number of patches: " + str(len(images_manager.patches)))
        


cdef void construct_patch(ImagesManager images_manager, list relevant_images, Feature feature):
    cdef double[:] f_coord = feature.coord
    cdef double[::1,:] f_camera_matrix = feature.img.camera_matrix()
    # check if cell occupied
    cdef Cell cell = feature.img.cell(
            j=int(f_coord[0] / CELL_SIZE),
            i=int(f_coord[1] / CELL_SIZE)) 
    if len(cell.t_patches) != 0:
        return

    cdef dict feat_data = {}
    cdef double[::1,:] fun_mat
    cdef double[:] p_c
    cdef double[:] depth_vector_1
    cdef double[:] depth_vector_2
    cdef double depth_1
    cdef double depth_2
    cdef double rel_depth

    def check_viable(Feature feat):
        fun_mat = images_manager.fundamental_matrix(feature.img, feat.img)
        p_c = triangulate(f_coord, feat.coord, f_camera_matrix,
                          feat.img.camera.camera_matrix, fun_mat)
        p_c = cdot(p_c, 1/p_c[3])

        if not visual_hull_check(p_c, images_manager):
            return False

        depth_vector_1 = vsub(feature.img.camera.optical_center, p_c)
        depth_vector_2 = vsub(feat.img.camera.optical_center, p_c)
        depth_1 = norm(depth_vector_1)
        depth_2 = norm(depth_vector_2)
        rel_depth = abs(depth_1 - depth_2)

        feat_data[feat] = (p_c, rel_depth)
        return True

    cdef list consistent_features = \
        match_epipolar_consistency(feature, images_manager,
                                   relevant_images, check_viable)
    
    cdef Patch patch = _init_patch(feature)

    def feature_key(f):
        return feat_data[f][1]

    consistent_features.sort(key=feature_key)

    cdef Feature c_f
    cdef double start_time
    cdef double elapsed_time
    for c_f in consistent_features:
        patch.center = feat_data[c_f][0]
        set_patch_images(patch, relevant_images, THRESHOLD1, THRESHOLD1)
        
        if len(patch.t_images) <= 1:
            continue

        optimize_similarity(patch, images_manager)

        set_patch_images(patch, relevant_images, THRESHOLD1, THRESHOLD2)

        if len(patch.t_images) >= T_THRESHOLD:
            register_patch(patch)
            images_manager.patches.append(patch)
            return


cdef void register_patch(Patch patch):
    cdef double[:] center = patch.center
    cdef double[:] img_coord
    cdef Cell cell
    cdef Image img

    # UGLY! needs to be refactored
    for img in patch.t_images:
        img_coord = matvmul(img.camera_matrix(), center)
        img_coord = cdot(img_coord, 1/img_coord[2])

        if img_coord[0] < 0 or img_coord[1] < 0:
            continue
        if img_coord[0] >= img.data.shape[1] or \
                img_coord[1] >= img.data.shape[0]:
            continue

        cell = img.cell(
            j=int(int(img_coord[0] / CELL_SIZE)),
            i=int(int(img_coord[1] / CELL_SIZE)))
        cell.t_patches.append(patch)

    for img in patch.f_images:
        img_coord = matvmul(img.camera_matrix(), center)
        img_coord = cdot(img_coord, 1/img_coord[2])

        if img_coord[0] < 0 or img_coord[1] < 0:
            continue
        if img_coord[0] >= img.data.shape[1] or \
                img_coord[1] >= img.data.shape[0]:
            continue

        cell = img.cell(
            j=int(int(img_coord[0] / CELL_SIZE)),
            i=int(int(img_coord[1] / CELL_SIZE)))
        cell.f_patches.append(patch)



cdef Patch _init_patch(Feature feat):
    cdef double[:] optical_center = feat.img.optical_center()
    cdef double[:] center = matvmul(feat.img.pinv(), feat.coord)
    center = cdot(center, 1.0/center[3])
    cdef double[:] normal = vsub(optical_center, center)
    normal = cdot(normal, 1/norm(normal))
    return Patch(r_image=feat.img, cell=feat.cell, normal=normal,
                 center=center)


cdef list match_epipolar_consistency(Feature feature,
                                     ImagesManager img_manager,
                                     list images,
                                     check_viable):
    cdef list consistent_features = []
    cdef Image feat_img = feature.img
    cdef list features

    cdef double[::1,:] fun_mat
    cdef double[:] epi_line
    cdef double max_dist
    cdef double epi_dist

    for img in images:
        if img == feat_img:
            continue

        if feature.feature_type == DOG:
            features = img.dog_features
        elif feature.feature_type == HARRIS:
            features = img.harris_features

        fun_mat = img_manager.fundamental_matrix(feat_img, img)
        epi_line = matvmul(fun_mat, feature.coord)

        max_dist = \
            CLEARANCE * sqrt(epi_line[0] ** 2 + epi_line[1] ** 2)

        for feat in features:
            epi_dist = abs(vdot(feat.coord, epi_line))
            if epi_dist <= max_dist and check_viable(feat):
                consistent_features.append(feat)

    return consistent_features



cpdef main(ImagesManager manager):
    perform_matching(manager)

    file = open("out.ply", "w")
    file.write("ply \n" +
               "format ascii 1.0 \n" +
               "element vertex " + str(len(manager.patches)) + "\n" +
               "property float x \n" +
               "property float y \n" +
               "property float z \n" +
               "property float nx \n" +
               "property float ny \n" +
               "property float nz \n" +
               "end_header \n")

    for p in manager.patches:
        file.write(str(p.center[0]) + " " + str(p.center[1]) + " " + str(
            p.center[2]) + " " +
                   str(p.normal[0]) + " " + str(p.normal[1]) + " " + str(
            p.normal[2]) + '\n')
    file.close()

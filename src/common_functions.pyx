import pybobyqa
from constants cimport *
from data_structs cimport *
from my_lapack cimport *
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport abs
from libc.math cimport acos
from libc.math cimport atan2
cimport cython


cpdef list get_relevant_images(Image img, ImagesManager img_manager):
    cdef list relevant_images = []
    cdef double[:] optical_axis_1, optical_axis_2
    cdef Image img_tmp
    cdef double cos_angle
    optical_axis_1 = img.optical_axis()
    for img_tmp in img_manager.images:
        if img_tmp == img:
            continue
        optical_axis_2 = img_tmp.optical_axis()
        cos_angle =  vdot(optical_axis_1, optical_axis_2)
        # if angle < max angle
        if cos_angle > COS_MAX_ANGLE:
            relevant_images.append(img_tmp)

    return relevant_images


cdef double[::1,:] b = np.array(np.vstack([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]).T, order='F')


cpdef list get_patch_vectors(Patch patch):
    """
    calculates the vectors (right, up) on the patch plane in world space
    so that moving along side one of them corresponds to moving a unit
    in the same direction in the reference image. in other words,
    the projections of (right, up) are (1, 0, 0) and (0, 1, 0) respectively
    :return: right, up
    """
    cdef double[::1,:] p = patch.r_image.camera_matrix()
    cdef double[::1,:] a = np.empty((p.shape[0] + 1, p.shape[1]), order='F')
    a[0:p.shape[0]] = p
    a[p.shape[0]] = patch.normal
    cdef double[::1,:] solution = solve2(a, b.copy_fortran())
    cdef double d = matvmul(p, patch.center)[2]
    return [cdot(solution[:, 0], d), cdot(solution[:, 1], d)]


cpdef double[:] triangulate(double[:] feat1_coord,
                           double[:] feat2_coord,
                           double[::1,:] proj1,
                           double[::1,:] proj2,
                           double[::1,:] fun_mat):
    cdef double[:] line = matvmul(fun_mat, feat1_coord)
    cdef double[:] line_perp = \
            np.array([-line[1], line[0], (line[1] * feat2_coord[0] -
                     line[0] * feat2_coord[1]) / feat2_coord[2]])
    
    cdef double[:] tmp = matvmul(proj2.T.copy_fortran(), line_perp)
    cdef double[:,:] a = np.vstack([proj1, np.dot(proj2.T, line_perp)])
    cdef double[::1,:] a_fortran = a.copy_fortran()

    cdef double[:] _b = np.array([feat1_coord[0], feat1_coord[1], feat1_coord[2], 0])
    return solve(a_fortran, _b)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef double correlation_coefficient(int[:,:,:] cell1, int[:,:,:] cell2):
    """
    calculates normlized cross correlation for two cells
    @params: source and destination cells
    @return: normlized cross correlation of (cell1, cell2)
    """
    cdef double cell1_mean = 0
    cdef double cell2_mean = 0
    cdef int x,y,z
    for x in range(cell1.shape[0]):
        for y in range(cell1.shape[1]):
            for z in range(cell1.shape[2]):
                cell1_mean += cell1[x,y,z]
                cell2_mean += cell2[x,y,z]
    cdef int tot = cell1.shape[0] * cell1.shape[1] * cell1.shape[2] 
    cell1_mean /= tot
    cell2_mean /= tot
    cdef double product = 0, std1 = 0, std2 = 0
    for x in range(cell1.shape[0]):
        for y in range(cell1.shape[1]):
            for z in range(cell1.shape[2]):
                product += (cell1[x,y,z] - cell1_mean) * (cell2[x,y,z] - cell2_mean) 
                std1 += (cell1[x,y,z] - cell1_mean) ** 2
                std2 += (cell2[x,y,z] - cell2_mean) ** 2

    cdef double stds = std1 * std2
    if stds == 0:
        return 0
    else:
        product /= sqrt(stds)
        return product


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef double ncc(Image img, Patch p, double[:] right, double[:] up, int[:,:,:] source_cell):
    """
    takes source patch, traverse around cell center and project
    each point on the destination image
    accumaltes pixel values in auxiliary cell
    and computes the ncc between the aux cell and the cell of the source patch
    """
    cdef double[:] img_center = matvmul(img.camera_matrix(), p.center)
    cdef float step = (PATCH_GRID_SIZE - 1) / 2.0
    cdef double[:] right_img = cdot_nocp(matvmul(img.camera_matrix(), right),  1.0 / img_center[2])
    cdef double[:] up_img = cdot_nocp(matvmul(img.camera_matrix(), up), 1.0 / img_center[2])
    img_center = cdot_nocp(img_center, 1.0 / img_center[2])

    cdef double[:] top_left = vsub(img_center, cdot_nocp(vadd(right_img, up_img), step))
    cdef int[:,:,:] p_cell = np.empty(shape=(PATCH_GRID_SIZE, PATCH_GRID_SIZE, 3), dtype=np.int32)

    cdef double z_x
    cdef double z_y
    cdef int y
    cdef int x
    for y in range(PATCH_GRID_SIZE):
        for x in range(PATCH_GRID_SIZE):
            z_x = top_left[0] + y * up_img[0] + x * right_img[0]
            z_y = top_left[1] + y * up_img[1] + x * right_img[1]
            p_cell[y, x] = img.get_pixel(x=z_x, y=z_y)

    return correlation_coefficient(source_cell, p_cell)


cpdef double similarity_function(Patch patch):
    """
    calculates accumlates ncc for source patch with t_images
    (truely visible images as said by essam)
    """
    cdef list right_up = get_patch_vectors(patch)
    cdef double[:] right = right_up[0]
    cdef double[:] up = right_up[1] 
    cdef int[:,:,:] source_cell = patch.source_cell

    cdef double accumlative_ncc = 0
    cdef Image img
    for img in patch.t_images:
        if img == patch.r_image:
            continue
        accumlative_ncc += ncc(img, patch, right, up, source_cell)
    accumlative_ncc /= (len(patch.t_images) - 1)
    return accumlative_ncc


cpdef double _optimize_similarity(params, args):
    """
    private helper function
    :param params: patch parameters to be optimized: depth, alpha, beta
    :param args: patch, unit_vector, optical_center
    :return: params that minimize the ncc
    """
    cdef double depth
    cdef double theta
    cdef double phi
    cdef ImagesManager images_manager
    cdef Patch patch
    cdef double[:] unit_vector
    cdef double scale_depth
    cdef double scale_theta

    depth, theta, phi = params
    patch, images_manager, unit_vector, scale_depth, scale_theta = args
    depth *= scale_depth
    theta *= scale_theta
    phi *= scale_theta

    patch.center = vadd(patch.r_image.optical_center(), cdot(unit_vector, depth)) 
    patch.normal = np.array([sin(theta) * cos(phi),
                             sin(theta) * sin(phi),
                             cos(theta), 0])

    if not visual_hull_check(patch.center, images_manager):
        return 1000

    return -similarity_function(patch)


cdef void optimize_similarity(Patch patch, ImagesManager images_manager):
    """
    runs optimization routine to find the patch params that maximize ncc
    :param patch: the patch to be optimized
    :param images_manager: needed to compute sil score
    """
    cdef double[:] optical_center
    cdef double[:] patch_center
    cdef double[:] depth_vector
    cdef double[:] unit_vector
    cdef double depth
    cdef double theta
    cdef double phi

    optical_center = patch.r_image.optical_center()
    patch_center = patch.center

    depth_vector = vsub(patch_center, optical_center)
    depth = norm(depth_vector)
    theta = acos(patch.normal[2])
    phi = atan2(patch.normal[1], patch.normal[0])
    unit_vector = cdot(depth_vector, 1/depth)

    cdef double[:] tmp = vadd(patch_center, unit_vector)
    cdef double scale_depth = 0
    cdef Image img
    cdef double[:] coord
    cdef double[:] coord2
    for img in patch.t_images:
        if img == patch.r_image:
            continue
        coord = matvmul(img.camera_matrix(), patch_center)
        coord = cdot(coord, 1/coord[-1])
        coord2 = matvmul(img.camera_matrix(), tmp)
        coord2 = cdot(coord2, 1/coord2[-1])
        scale_depth += norm(np.subtract(coord2, coord))

    scale_depth /= (len(patch.t_images) - 1)
    scale_depth = 2 / scale_depth
    scale_theta = np.pi / 48

    depth /= scale_depth
    theta /= scale_theta
    phi /= scale_theta

    cdef tuple args = \
        (patch, images_manager, unit_vector, scale_depth, scale_theta)
    cdef np.ndarray params = np.array([depth, theta, phi])

    def objective(x):
        return _optimize_similarity(x, args)

    cdef np.ndarray lower_bounds = np.array([float('-inf'), theta - (np.pi / 4) / scale_theta,
                             phi - (np.pi / 4) / scale_theta])
    cdef np.ndarray upper_bounds = np.array([float('inf'), theta + (np.pi / 4) / scale_theta,
                             phi + (np.pi / 4) / scale_theta])

    pybobyqa.solve(_optimize_similarity, params, objective,
                          (lower_bounds, upper_bounds))


cpdef bint visual_hull_check(double[:] center, ImagesManager images_manager):
    cdef Image img
    cdef double[:] img_center
    for img in images_manager.images:
        img_center = matvmul(img.camera_matrix(), center)
        img_center = cdot_nocp(img_center, 1.0/img_center[2])
        if not img.check_in_image(x=img_center[0], y=img_center[1]):
            continue
        val = img.silhouette[int(round(img_center[1])),
                             int(round(img_center[0]))]
        if val == 0:
            return False
    return True


cpdef void set_patch_images(Patch patch, list images, double alpha_s, double alpha_t):
    cdef double[:] right
    cdef double[:] up
    right, up = get_patch_vectors(patch)
    
    patch.t_images = [patch.r_image]
    patch.s_images = [patch.r_image]    

    cdef double[:] depth_vector
    cdef Image img
    cdef double ncc_score

    for img in images:
        depth_vector = vsub(img.optical_center(), patch.center)
        if vdot(depth_vector, patch.normal) <= 0:
            continue
        ncc_score = ncc(img, patch, right, up, patch.source_cell) 
        if ncc_score >= alpha_s:
            patch.s_images.append(img)
            if ncc_score >= alpha_t:
                patch.t_images.append(img)
            else:
                patch.f_images.append(img)
    
from data_structs cimport *


cpdef list get_relevant_images(Image img, ImagesManager img_manager)

cpdef list get_patch_vectors(Patch patch)

cpdef double[:] triangulate(double[:] feat1_coord,
                           double[:] feat2_coord,
                           double[::1,:] proj1,
                           double[::1,:] proj2,
                           double[::1,:] fun_mat)

cpdef double correlation_coefficient(int[:,:,:] cell1, int[:,:,:] cell2)

cpdef double ncc(Image img, Patch p, double[:] right, double[:] up, int[:,:,:] source_cell)

cpdef double similarity_function(Patch patch)

cpdef bint visual_hull_check(double[:] center, ImagesManager images_manager)

cpdef void set_patch_t_images(Patch patch, list images, double alpha)

cdef void optimize_similarity(Patch patch, ImagesManager images_manager)

cpdef double _optimize_similarity(params, args)

cpdef int DOG
cpdef int HARRIS


cdef class Camera:
    cdef public double[::1,:] camera_matrix
    cdef public double[::1,:] pinv
    cdef public double[:] optical_axis
    cdef public double[:] optical_center 


cdef class Feature:
    cdef public double[:] coord
    cdef public int y
    cdef public int x
    cdef public int feature_type
    cdef public Cell cell 
    cdef public Image img 


cdef class Cell:
    cdef public list t_patches
    cdef public list f_patches
    cdef public int x_initial
    cdef public int y_initial
    cdef public double depth
    cdef public Image img


cdef class Patch:
    cdef public list t_images
    cdef public list s_images
    cdef public list f_images
    cdef public Image r_image
    cdef public double[:] normal
    cdef public double[:] center
    cdef public Cell cell
    cdef public int[:] image_center
    cdef public int[:,:,:] source_cell


cdef class Image:
    cdef public int[:,:,:] data
    cdef public int[:,:] silhouette
    cdef public Camera camera
    cdef public int x_cells
    cdef public int y_cells
    cdef public list dog_features
    cdef public list harris_features
    cdef public list cells 
    cdef public int[:] _zeros

    cdef public Cell cell(self, int i, int j)
    cdef public double[::1,:] camera_matrix(self)
    cdef public void add_feature(self, Feature feat)
    cdef public double[::1,:] pinv(self)
    cdef public double[:] optical_center(self)
    cdef public double[:] optical_axis(self)
    cdef public bint check_in_image(self, double y, double x)
    cdef public int[:] get_pixel(self, double y, double x)
    cdef public int[:] nearest_neighbor(self, double y, double x)


cdef class ImagesManager:
    cdef public list images
    cdef public list patches 
    cdef public dict _fundamental_matrices
    cdef public dict _camera_distances

    cdef public double[::1,:] fundamental_matrix(self, Image img1, Image img2)
    cdef public double camera_distance(self, Camera cam1, Camera cam2)


cdef public double[::1,:] _skew_form(double[:] e)
cdef public double[:] epipole(Image img1, Image img2)

    

import feature_detector
import optimization_utilities
import numpy as np
from images_manager import *
from feature_detector import *
from optimization_utilities import *
import threading
<<<<<<< HEAD

=======
import time
>>>>>>> 2cfee8a1c1c398ca89b4b5c5a4f84eebdbc6d987
def perform_matching(images_manager):
    print("begin matching...")
    print("detecting features...")
    for img in images_manager:
        feature_detector.detect_features(img)
    print("done detecting features!")
    print("begin feature matching...")
    i = 1
    for img in images_manager:
        while( len( threading.enumerate()) >1):
            time.sleep(3)
            print("threading ",len( threading.enumerate()) )
        print("start img " + str(i))
        patches_num = len(images_manager.patches)
<<<<<<< HEAD
        #for feature in (img.dog_features + img.harris_features):
            #construct_patch(images_manager,feature)
        features =img.dog_features + img.harris_features
        t1 = threading.Thread(target=process_threads, args=(features[0:400],images_manager,))
        t1.start() 
        t2= threading.Thread(target=process_threads, args=(features[400:],images_manager,))
        t2.start() 
=======
        features = img.dog_features + img.harris_features
        #for feature in (img.dog_features + img.harris_features):
            #construct_patch(images_manager,feature)
>>>>>>> 2cfee8a1c1c398ca89b4b5c5a4f84eebdbc6d987

        #patches_num = -patches_num + len(images_manager.patches)(
        t1 = threading.Thread(target=process_features,args=(features[:400],images_manager))
        t2 = threading.Thread(target=process_features,args=(features[400:],images_manager))
        t1.start()
        t2.start()
        print("done img " + str(i) + " " + str(patches_num) + \
            " created succesfully!")
    while(len(threading.enumerate())>1):
        time.sleep(3)
        print('waiting ..')

def construct_patch(images_manager,feature,gamma=3   ):
    consistent_features =  optimization_utilities.match_epipolar_consistency(\
     feature, images_manager)

    feature_depth = {}
    f_coord = np.array( [feature.x,feature.y,1] )
    f_camera_matrix = feature.img.camera_matrix()
    f_optical_matrix = feature.img.optical_center()
    for c_f in consistent_features:
        fun_mat = images_manager.fundamental_matrix(feature.img,c_f.img)
        c_f_coord = np.array( [c_f.x,c_f.y,1] )
        c_f_camera_matrix = c_f.img.camera_matrix()

        p_c = optimization_utilities.triangulate( c_f_coord, f_coord, f_camera_matrix,\
         c_f_camera_matrix, fun_mat)

        depth = np.linalg.norm(p_c - f_optical_matrix)
        feature_depth[c_f] = [depth,p_c]

    get_feature_depth = lambda f : feature_depth[f][0]
    consistent_features.sort(key=get_feature_depth)

    for c_f in consistent_features:
        patch =  _construct_candidate_patch(feature,feature_depth[c_f][1],images_manager)
        if(len(patch.t_images) >= gamma):
            images_manager.patches.append(patch)
            break



def _construct_candidate_patch(feat,c,images_manager,alpha=0.5):
     normal = (c-feat.img.camera.optical_center)
     normal = normal/np.linalg.norm(normal)
     p =  Patch(r_image=feat.img, cell=feat.cell,normal=normal, center=c)
     optimization_utilities.set_patch_t_images(p,images_manager,alpha)
     if(len(p.t_images) != 0 ):
        optimization_utilities.optimize_similarity(p)

     p.t_images = []
     optimization_utilities.set_patch_t_images(p,images_manager,alpha * 1.1)
     return p

<<<<<<< HEAD
def process_threads(features,images_manager):
     print("starting new thread")
     for feature in (features):
            construct_patch(images_manager,feature)
               
=======
def process_features(features,images_manager):
    print("running new thread")
    for feature in features:
        construct_patch(images_manager,feature)
>>>>>>> 2cfee8a1c1c398ca89b4b5c5a4f84eebdbc6d987

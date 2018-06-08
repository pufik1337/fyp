import numpy as np
from . import inout
from . import misc

def transform_pose_mat(pose_mat):
    # separatest the rotational and positional components of the pose 
    # and implements the rodriguez exponential map in SO(3) to obtain a 3-parameter
    # representation of the rotation;
    # returns the 4D pose vector
    rot_mat = pose_mat[0:3, :]
    trans_vec = pose_mat[3, :]

    cos_phi = (np.trace(rot_mat) - 1.0) / 2.0
    cos_phi = np.minimum(np.maximum(cos_phi, -0.999999), 0.999999) #needed to prevent mapping to nan
    phi = np.arccos(cos_phi)
    rot_mat = phi*(rot_mat - np.transpose(rot_mat))/(2*np.sin(phi))

    rot_vec = [rot_mat[2][1], -rot_mat[2][0], rot_mat[1][0]]
    depth = trans_vec[2]
    
    return np.hstack((rot_vec, depth))

def transform_poses(poses):
    for pose_mat in poses[0]:
        pose_mat = transform_pose_mat(pose_mat)
    return poses

def recover_6d_pose(pose_vec, bbox, pose_mean, pose_stddev):
    # implements the rodriguez exponential map to recover the 3x3 rotation matrix and 
    # calculates the translation vector using the regressed depth and bounding box
    # inputs: pose_vec - a 3x1 vector in so(3)
    # bbox: a 4x1 vector in the form of (ymin, xmin, ymax, xmax)

    pose_vec = np.add(np.multiply(pose_vec, pose_stddev), pose_mean)

    w = np.zeros([3, 3])
    w[2, 1] = pose_vec[0]
    w[1, 2] = -pose_vec[0]
    w[2, 0] = -pose_vec[1]
    w[0, 2] = pose_vec[1]
    w[1, 0] = pose_vec[2]
    w[0, 1] = -pose_vec[2]
    w_magnitude = np.linalg.norm(pose_vec[0:3])

    rot_mat = np.identity(3) + (np.sin(w_magnitude)/w_magnitude)*w + ((1 - np.cos(w_magnitude))/(w_magnitude**2))*(np.matmul(w, w))

    # parameters of the intrinsic camera matrix
    fx, cx, fy, cy = 571.9737, 319.5, 571.0073, 239.5

    # get rid of the magic number later
    tz = pose_vec[3]

    ux = (bbox[1] + bbox[3])/2.0
    uy = (bbox[0] + bbox[2])/2.0

    tx = (ux - cx)*tz/fx
    ty = (uy - cy)*tz/fy

    t_vec = [tx, ty, tz]

    return np.asarray(rot_mat), np.asarray(t_vec)

def load_models(model_path, n_fg_class):
    models = {}
    for classId in range(1, n_fg_class + 1):
        models[classId] = inout.load_ply(model_path + "/obj_" + str(classId).zfill(2) + ".ply")

    return models

def get_model_diameters(models_dict):
    diameters = {}
    print("getting model diameters: ")
    for key, model in models_dict.items():
        diameters[key] = misc.calc_pts_diameter(model['pts'])
        print("diameter of ", key, diameters[key])
    return diameters

#models = load_models("/home/ubuntu/fyp/models", 6)
#print(models[1])
#diameters = get_model_diameters(models)
#print(diameters)

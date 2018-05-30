# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016

import math
import numpy as np
from scipy import spatial
from . import misc


def add(R_est, t_est, R_gt, t_gt, model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = misc.transform_pts_Rt(model['pts'], R_est, t_est)
    pts_gt = misc.transform_pts_Rt(model['pts'], R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def add_metric(R_est, t_est, R_gt, t_gt, model, diameter):
    avg_dist = add(R_est, t_est, R_gt, t_gt, model)

    diameters = {1:125.64433088293319, 2:136.6157742173282, 3:235.60199312418814, 4:220.63983688128488, 5:254.5407041137429, 6:188.56383670982623}

    print("average dist:", avg_dist)
    print("diameter of the model: ", diameter)

    if avg_dist <= 0.1*diameter:
        return 1.0
    else:
        return 0.0

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

def five_by_five_metric(R_est, t_est, R_gt, t_gt):
    rot_error = re(R_est, R_gt)
    trans_error = te(t_est, t_gt)

    print("rotation error:", rot_error)
    print("translation error:", trans_error)

    if rot_error <= 5.0 and trans_error <=  50.0:
        return 1.0
    else:
        return 0.0



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

    print("average dist:", avg_dist)
    print("diameter of the model: ", diameter)

    if avg_dist <= 0.1*diameter:
        return 1.0
    else:
        return 0.0


import os

import ipdb
import matplotlib
import pickle
import gc
from tqdm import tqdm

from utils.config import opt
from data.dataset import TestDataset, inverse_normalize
from torch.autograd import Variable
from torch.utils import data as data_
from utils import array_tool as at
#from utils import pose_tool as pt
from utils.vis_tool import visdom_bbox, vis_bbox
from utils.eval_tool import eval_network_tejani
from utils.vis_tool import Visualizer
from utils import pose_error as pe
from utils.pose_tool import load_models, recover_6d_pose
from model.utils.bbox_tools import bbox_iou
import numpy as np

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def draw_pose(R, t, model, mode, img_size, accepted, blend_factor=0.05):
    K = 1.25*np.asarray([571.9737, 0.0, 319.5, 0.0, 571.0073, 239.5, 0.0, 0.0, 1.0]).reshape(3, 3)
    img = pe.render(model, img_size, K, R, t, mode=mode)
    if mode == 'depth':
        return np.stack((img, img, img))
    elif mode == 'rgb':
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        return np.stack((abs((accepted - 1.0 + blend_factor))*(0.34*green + 0.33*red + 0.33*blue), 
                         abs(accepted - blend_factor)*(0.34*green + 0.33*red + 0.33*blue), 
                         0.25*blue))

def blend_pose(original_img, pose_img):
    avg = np.mean(pose_img, axis=0)
    mask = avg > 0
    return pose_img + 0.65*np.logical_not(mask)*original_img
    #return pose_img
    #return(255*mask)

def eval(dataloader, pose_mean, pose_stddev, test_metric='add'):
    f = open('store_inference_rgbd.pckl', 'rb')
    vis = Visualizer(env=opt.env)
    models = load_models("/home/pufik/fyp/tejani/models", 6)
    diameters = [125.64433088293319, 136.6157742173282, 235.60199312418814, 220.63983688128488, 254.5407041137429, 188.56383670982623]
    K = np.asarray([571.9737, 0.0, 319.5, 0.0, 571.0073, 239.5, 0.0, 0.0, 1.0]).reshape(3, 3)
    # pred_bboxes, pred_poses, pred_labels, pred_scores = list(), list(), list(), list()
    # gt_bboxes, gt_poses, gt_labels, gt_difficults = list(), list(), list(), list()
    [pred_bboxes, pred_poses, pred_labels, pred_scores,
    gt_bboxes, gt_poses, gt_labels, _, pose_mean, pose_stddev, _, gt_difficults] = pickle.load(f)
    for ii, (rgb_imgs, depth_imgs, sizes, gt_bboxes_, gt_poses_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        #print(ii)
        if(ii%20 == 0):
            rgb_img = inverse_normalize(at.tonumpy(rgb_imgs[0])).reshape(600, 800, 3)
            rgb_img = rgb_img.reshape(3, 600, 800)
            pred_bb = pred_bboxes[ii]
            pred_label = pred_labels[ii]
            pred_score = pred_scores[ii]
            #rgb_img = at.tonumpy(rgb_imgs[0])
            print(rgb_img.shape)
            gt_pose_img = np.zeros((3, 600, 800))
            angle_errors = []
            pose_errors = []
            ious = []
            pred_pose_img = np.zeros((3, 600, 800))
            #for pred_pose_item, gt_pose_item, pred_bbox_item, gt_label_item in zip(pred_poses[ii], gt_poses[ii], pred_bboxes[ii], gt_labels[ii]):
            #for pred_pose, gt_pose, pred_label, gt_label, pred_bbox, gt_bbox in zip(pred_poses, gt_poses, pred_labels, gt_labels, pred_bboxes, gt_bboxes):
            iou = bbox_iou(pred_bboxes[ii], gt_bboxes[ii])
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < 0.5] = -1

            for pred_idx in range(iou.shape[0]):
                gt_idx = gt_index[pred_idx]

                pred_pose_item = pred_poses[ii][pred_idx]
                gt_pose_item = gt_poses[ii][gt_idx]
                gt_label_item = gt_labels[ii][gt_idx]
                pred_bbox_item = pred_bboxes[ii][pred_idx]

                pred_R, pred_t = recover_6d_pose(pred_pose_item, pred_bbox_item, pose_mean, pose_stddev)
                gt_R, gt_t = np.asarray(gt_pose_item[0:3, :]), np.asarray(gt_pose_item[3, :])
                #accepted = pe.five_by_five_metric(pred_R, pred_t, gt_R, gt_t)
                #accepted = pe.add_metric(pred_R, pred_t, gt_R, gt_t, models[gt_label_item + 1], diameters[gt_label_item])
                iou = pe.iou(pred_R, pred_t, gt_R, gt_t, models[gt_label_item + 1], (800, 600), K)
                #accepted = pe.five_by_five_metric(pred_R, pred_t, gt_R, gt_t)
                pose_errors += [bool(iou >= 0.5)]
                #angle_errors += [pe.re(pred_R, gt_R)]
                ious += [iou]
                gt_pose_img += draw_pose(np.asarray(gt_R), np.asarray(gt_t), models[gt_label_item + 1], 'rgb', (800, 600), 1.0)
                pred_pose_img += draw_pose(np.asarray(pred_R), np.asarray(pred_t), models[gt_label_item + 1], 'rgb', (800, 600), bool(iou >= 0.5))
            print("pose error: ", pose_errors)
            pred_img_rgb = visdom_bbox(blend_pose(rgb_img, pred_pose_img),
                                    at.tonumpy(pred_bb),
                                    at.tonumpy(pred_label),
                                    at.tonumpy(pred_score),
                                    iou=ious
                                    )
            gt_img_rgb = visdom_bbox(blend_pose(rgb_img, gt_pose_img), at.tonumpy(gt_bboxes_[0]), at.tonumpy(gt_labels_[0]))     
            #ax = vis.img('rgb_img', gt_img_rgb)
            vis.img('gt_img', gt_img_rgb)
            vis.img('pred_img', pred_img_rgb)
            vis.log(ii, 'iter')
            gc.collect()
            #vis.img('gt_pose_img', 0.7*gt_pose_img+0.3*rgb_img)
            #vis.img('pred_pose_img', 0.7*pred_pose_img+0.3*rgb_img)
                

    #     sizes = [sizes[0][0], sizes[1][0]]
    #     pred_bboxes_, pred_poses_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
    #     gt_bboxes += list(gt_bboxes_.numpy())
    #     gt_poses += list(gt_poses_.numpy())
    #     gt_labels += list(gt_labels_.numpy())
    #     gt_difficults += list(gt_difficults_.numpy())
    #     pred_bboxes += pred_bboxes_
    #     pred_poses += pred_poses_
    #     pred_labels += pred_labels_
    #     pred_scores += pred_scores_
    f.close
    #print("pred_poses: ", pred_poses)
    #print("gt poses: ", gt_poses)
    print("file closed")
    result = eval_network_tejani(
        pred_bboxes, pred_poses, pred_labels, pred_scores,
        gt_bboxes, gt_poses, gt_labels,"/home/pufik/fyp/tejani/models", pose_mean, pose_stddev, 6, gt_difficults,
        use_07_metric=True, test_metric=test_metric)
    return result


def test(**kwargs):
    opt._parse(kwargs)
    testset = TestDataset(opt)
    pose_mean = testset.pose_mean
    pose_stddev = testset.pose_stddev
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    eval_result = eval(test_dataloader, pose_mean, pose_stddev, test_metric=opt.test_metric)
    print("Evaluation result : ", eval_result)

if __name__ == '__main__':
    import fire

    fire.Fire()

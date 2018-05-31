import os

import ipdb
import matplotlib
import pickle
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from torch.autograd import Variable
from torch.utils import data as data_
from utils import array_tool as at
from utils import pose_tool as pt
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_network_tejani

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(test_metric='add'):
    f = open('/home/ubuntu/store_inference.pckl', 'rb')

    # pred_bboxes, pred_poses, pred_labels, pred_scores = list(), list(), list(), list()
    # gt_bboxes, gt_poses, gt_labels, gt_difficults = list(), list(), list(), list()
    # for ii, (imgs, sizes, gt_bboxes_, gt_poses_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
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
    #     if ii == test_num: break
    [pred_bboxes, pred_poses, pred_labels, pred_scores,
        gt_bboxes, gt_poses, gt_labels, _, pose_mean, pose_stddev, _, gt_difficults] = pickle.load(f)
    f.close
    #print("pred_poses: ", pred_poses)
    print("gt poses: ", gt_poses)
    print("file closed")
    result = eval_network_tejani(
        pred_bboxes, pred_poses, pred_labels, pred_scores,
        gt_bboxes, gt_poses, gt_labels,"/home/ubuntu/fyp/models", pose_mean, pose_stddev, 6, gt_difficults,
        use_07_metric=True, test_metric=test_metric)
    return result


def test(**kwargs):
    opt._parse(kwargs)
    eval_result = eval(test_metric=opt.test_metric)
    print("Evaluation result : ", eval_result)

if __name__ == '__main__':
    import fire

    fire.Fire()

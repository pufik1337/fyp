import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import Deep6DRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from pose_trainer import FasterRCNNPoseTrainer
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


def eval(dataloader, faster_rcnn, pose_mean, pose_stddev, test_num=10000, test_metric='add'):
    pred_bboxes, pred_poses, pred_labels, pred_scores = list(), list(), list(), list()
    gt_bboxes, gt_poses, gt_labels, gt_difficults = list(), list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_poses_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_poses_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_poses += list(gt_poses_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_poses += pred_poses_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_network_tejani(
        pred_bboxes, pred_poses, pred_labels, pred_scores,
        gt_bboxes, gt_poses, gt_labels,"/home/ubuntu/fyp/models", pose_mean, pose_stddev, 6, gt_difficults,
        use_07_metric=True)
    return result


def test(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    pose_mean = dataset.pose_mean
    pose_stddev = dataset.pose_stddev

    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = Deep6DRCNNVGG16(n_fg_class=6)
    print('model construct completed')
    trainer = FasterRCNNPoseTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    eval_result = eval(test_dataloader, faster_rcnn, pose_mean, pose_stddev, test_num=opt.test_num, test_metric=opt.test_metric)
    print("Evaluation result : ", eval_result)

if __name__ == '__main__':
    import fire

    fire.Fire()

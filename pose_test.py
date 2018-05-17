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
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_network_tejani

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def eval(dataloader, faster_rcnn, trainer, pose_mean, pose_stddev, test_num=1):
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
        if ii%100 == 0:
                ori_img_ = inverse_normalize(at.tonumpy(imgs[0]))
                resized_gt_bbox = resize_bbox(gt_bboxes[0], (600, 800), (480, 640))
                resized_pred_bbox = resize_bbox(pred_bboxes_[0], (600, 800), (480, 640))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(resized_gt_bbox),
                                     at.tonumpy(gt_labels_[0]))
                trainer.vis.img('gt_img', gt_img)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(resized_pred_bbox),
                                       at.tonumpy(pred_labels_[0]).reshape(-1),
                                       at.tonumpy(pred_scores_[0]))
                trainer.vis.img('pred_img', pred_img)
    result = eval_network_tejani(
        pred_bboxes, pred_poses, pred_labels, pred_scores,
        gt_bboxes, gt_poses, gt_labels,"/home/ubuntu/fyp/models", pose_mean, pose_stddev, 6, gt_difficults,
        use_07_metric=True)
    return result


def test(**kwargs):
    print("TEST TEST TEST TEST")
    opt._parse(kwargs)
    print(opt)
    dataset = Dataset(opt)
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
    print('Evaluating model:')
    eval_result = eval(test_dataloader, faster_rcnn, trainer, testset.pose_mean, testset.pose_stddev, test_num=opt.test_num)
    print(eval_result)

if __name__ == '__main__':
    import fire
    print("YO YO YO YO")
    fire.Fire()

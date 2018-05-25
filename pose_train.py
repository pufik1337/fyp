import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import Deep6DRCNNVGG16_RGBD
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


def eval(dataloader, faster_rcnn, pose_mean, pose_stddev, test_num=10000):
    pred_bboxes, pred_poses, pred_labels, pred_scores = list(), list(), list(), list()
    gt_bboxes, gt_poses, gt_labels, gt_difficults = list(), list(), list(), list()
    for ii, (rgb_imgs, depth_imgs, sizes, gt_bboxes_, gt_poses_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_poses_, pred_labels_, pred_scores_ = faster_rcnn.predict(rgb_imgs, depth_imgs, [sizes])
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


def train(**kwargs):
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
    faster_rcnn = Deep6DRCNNVGG16_RGBD(n_fg_class=6)
    print("faster_rcnn: ", faster_rcnn)
    print('model construct completed')
    trainer = FasterRCNNPoseTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        #print("in epoch")
        trainer.reset_meters()
        #print("just before enumerate")
        for ii, (rgb_img, depth_img, bbox_, pose_, label_, scale) in tqdm(enumerate(dataloader)):
            #print("inside enumerate")
            scale = at.scalar(scale)
            #print("pose :", pose_)
            color_img, depth_img, bbox, pose, label = rgb_img.cuda().float(), depth_img.cuda().float(), bbox_.cuda(), pose_.cuda(), label_.cuda()
            color_img, depth_img, bbox, pose, label = Variable(color_img), Variable(depth_img), Variable(bbox), Variable(pose), Variable(label)
            trainer.train_step(color_img, depth_img, bbox, pose, label, scale)
            
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_rgb_ = inverse_normalize(at.tonumpy(color_img[0]))
                ori_img_depth_ = inverse_normalize(at.tonumpy(depth_img[0]))
                gt_img_rgb = visdom_bbox(ori_img_rgb_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                gt_img_depth = visdom_bbox(ori_img_depth_, at.tonumpy(bbox_[0]), at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img_rgb)
                trainer.vis.img('gt_img_depth', gt_img_depth)

                # plot predicti bboxes
                _bboxes, _poses, _labels, _scores = trainer.faster_rcnn.predict([ori_img_rgb_], [ori_img_depth_], visualize=True)
                print("Predicted : \n")
                print(_poses)
                print("Ground truth : \n")
                print(pose)
                pred_img = visdom_bbox(ori_img_rgb_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, pose_mean, pose_stddev, test_num=opt.test_num)
        print("Epoch %d evaluation result : ", epoch)
        print(eval_result)

        if eval_result['mean_pose_add'] > best_map and eval_result['mean_pose_add'] > 0.01:
            best_map = eval_result['mean_pose_add']
            best_path = trainer.save(best_map=best_map)
        if epoch == 5:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        trainer.vis.plot('test_map', eval_result['map'])
        trainer.vis.plot('test_pose_add', eval_result['mean_pose_add'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        if epoch == 15: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

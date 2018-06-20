import os
import xml.etree.ElementTree as ET
import yaml
import time
import math

import numpy as np

from .util import read_image, load_depth2


class TejaniBboxDataset:
    """Bounding box dataset for Tejani

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        self.SEQ_COUNTS = [265, 414, 543, 410, 95, 340]
        self.TEST_COUNTS = []
        self.TRAINVAL_COUNTS = []
        self.TRAINVAL_IDS = {}
        self.TEST_IDS = {}

        for i in range(len(self.SEQ_COUNTS)):
            np.random.seed(0)
            scrambled_ids = np.random.permutation(self.SEQ_COUNTS[i])
            boundary = round(0.3*self.SEQ_COUNTS[i])
            self.TRAINVAL_COUNTS += [boundary]
            self.TEST_COUNTS += [self.SEQ_COUNTS[i] - boundary]
            self.TRAINVAL_IDS[i] = scrambled_ids[0:boundary]
            self.TEST_IDS[i] = scrambled_ids[boundary:self.SEQ_COUNTS[i]]

        self.split = split
        self.data_dir = data_dir
        self.label_names = TEJANI_BBOX_LABEL_NAMES
        self.return_difficult = return_difficult
        self.use_difficult = use_difficult

        if split == 'test':
            self.ids = range(sum(self.TEST_COUNTS))
            self.anno = {}
            for i in range(6):
                self.anno[i] = yaml.load(open(os.path.join(data_dir, '0' + str(i + 1), 'gt.yml')))
        elif split == 'trainval':
            self.ids = range(sum(self.TRAINVAL_COUNTS))
            self.anno = {}
            for i in range(6):
                self.anno[i] = yaml.load(open(os.path.join(data_dir, '0' + str(i + 1), 'gt.yml')))
        else:
            id_list_file = os.path.join(
                data_dir, 'train.txt')

            #self.ids = [id_.strip() for id_ in open(id_list_file)]
            self.ids = range(sum(self.TRAINVAL_COUNTS))

        pose_sum = np.empty([1, 4])
        for i in range(len(self)):
            pose_sum = np.vstack((self.get_example(i)[2], pose_sum))
        self.pose_mean = np.mean(pose_sum, axis=0)
        self.pose_stddev = np.std(pose_sum, axis=0)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i, mode='rgb', train=True, normalize=False):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        if self.split == 'test' or self.split == 'trainval':
            id_ = 0
            classId = 0
            while True:
                if self.split =='test':
                    if (i < sum(self.TEST_COUNTS[:classId + 1]) and i >= sum(self.TEST_COUNTS[:classId])):
                        id_ = i - sum(self.TEST_COUNTS[:classId])
                        id_ = self.TEST_IDS[classId][id_]
                        #print("Class id: ", classId)
                        break
                    classId += 1
                else:
                    if (i < sum(self.TRAINVAL_COUNTS[:classId + 1]) and i >= sum(self.TRAINVAL_COUNTS[:classId])):
                        id_ = i - sum(self.TRAINVAL_COUNTS[:classId])
                        id_ = self.TRAINVAL_IDS[classId][id_]
                        #print("Class id: ", classId)
                        break
                    classId += 1                    
            bbox = list()
            pose = list()
            label = list()
            difficult = list()
            for obj in self.anno[classId][id_]:
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                bndbox_anno = obj['obj_bb'].copy()
                rot_anno = np.array(obj['cam_R_m2c'].copy())
                trans_anno = np.array(obj['cam_t_m2c'].copy())
                rot_anno = rot_anno.reshape(3, 3)
                #trans_anno = trans_anno.reshape(1, 3)i
                if not train:
                    pose.append(np.vstack((rot_anno, trans_anno)))
                else:
                    cos_phi = (np.trace(rot_anno) - 1.0) / 2.0
                    cos_phi = np.minimum(np.maximum(cos_phi, -0.999999), 0.999999) #needed to prevent mapping to nan
                    phi = np.arccos(cos_phi)
                    rot_anno = phi*(rot_anno - np.transpose(rot_anno))/(2*np.sin(phi))
                    pose_vec = [rot_anno[2][1], -rot_anno[2][0], rot_anno[1][0], trans_anno[2]]
                    if normalize:
                        pose_vec = np.divide(np.add(pose_vec, -self.pose_mean), self.pose_stddev)
                    pose.append(pose_vec)
                difficult.append(0)
                # subtract 1 to make pixel indexes 0-based
                bndbox_anno[0] = obj['obj_bb'][1]
                bndbox_anno[1] = obj['obj_bb'][0]
                bndbox_anno[2] = obj['obj_bb'][3] + obj['obj_bb'][1]
                bndbox_anno[3] = obj['obj_bb'][2] + obj['obj_bb'][0]
                bbox.append(bndbox_anno)
                #pose.append(np.vstack((rot_anno, trans_anno)))
                #pose.append(pose_vec)
                name = obj['obj_id'] - 1
                assert(name == classId)
                label.append(name)
            #print (bbox, label)        
            bbox = np.stack(bbox).astype(np.float32)
            pose = np.stack(pose).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
            if mode == 'rgb':
                img_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), mode,  str(id_).zfill(4) + '.jpg')
                img = read_image(img_file, color=True)
            elif mode == 'depth':
                img_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), mode,  str(id_).zfill(4) + '.png')
                img = load_depth2(img_file)
                img = np.stack((img, img, img))
            elif mode == 'rgbd':
                color_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), 'rgb',  str(id_).zfill(4) + '.jpg')
                color_img = read_image(color_file, color=True)
                depth_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), 'depth',  str(id_).zfill(4) + '.png')
                depth_img = load_depth2(depth_file)
                depth_img = np.stack((depth_img, depth_img, depth_img))
                img = color_img, depth_img
            else:
                raise ValueError("Invalid Dataset Wrapper Mode: {}; must be one of (rgb, depth, rgbd)".format(mode))

            return img, bbox, pose, label, difficult
            
                    
        else:
            pass

    __getitem__ = get_example

    def shuffle(self):
        for key, value in self.TRAINVAL_IDS.items():
            self.TRAINVAL_IDS[key] = np.random.permutation(value)


TEJANI_BBOX_LABEL_NAMES = (
    'camera',
    'cup',
    'joystick',
    'juice',
    'milk',
    'shampoo')

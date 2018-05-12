import os
import xml.etree.ElementTree as ET
import yaml
import time
import math

import numpy as np

from util import read_image


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

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        if split == 'test':
            self.ids = range(sum(TEST_COUNTS))
            self.anno = {}
            for i in range(6):
                self.anno[i] = yaml.load(open(os.path.join(data_dir, '0' + str(i + 1), 'gt.yml')))
        elif split == 'trainval':
            self.ids = range(sum(TRAINVAL_COUNTS))
            self.anno = {}
            for i in range(6):
                self.anno[i] = yaml.load(open(os.path.join(data_dir, '0' + str(i + 1), 'gt.yml')))
        else:
            id_list_file = os.path.join(
                data_dir, 'train.txt')

            #self.ids = [id_.strip() for id_ in open(id_list_file)]
            self.ids = range(sum(TRAINVAL_COUNTS))
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = TEJANI_BBOX_LABEL_NAMES
        self.split = split

    def __len__(self):
        return len(self.ids)

    def get_example(self, i, mode='rgb'):
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
                    if (i < sum(TEST_COUNTS[:classId + 1]) and i >= sum(TEST_COUNTS[:classId])):
                        id_ = i - sum(TEST_COUNTS[:classId])
                        id_ = TEST_IDS[classId][id_]
                        #print("Class id: ", classId)
                        break
                    classId += 1
                else:
                    if (i < sum(TRAINVAL_COUNTS[:classId + 1]) and i >= sum(TRAINVAL_COUNTS[:classId])):
                        id_ = i - sum(TRAINVAL_COUNTS[:classId])
                        id_ = TRAINVAL_IDS[classId][id_]
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
                #trans_anno = trans_anno.reshape(1, 3)
                #cos_phi = (np.trace(rot_anno) - 1.0) / 2.0
                #cos_phi = np.minimum(np.maximum(cos_phi, -0.999999), 0.999999) #needed to prevent mapping to nan
                #phi = np.arccos(cos_phi)
                #rot_anno = phi*(rot_anno - np.transpose(rot_anno))/(2*np.sin(phi))
                #pose_vec = [rot_anno[2][1], -rot_anno[2][0], rot_anno[1][0], trans_anno[2]/1000.0]
                difficult.append(0)
                # subtract 1 to make pixel indexes 0-based
                bndbox_anno[0] = obj['obj_bb'][1]
                bndbox_anno[1] = obj['obj_bb'][0]
                bndbox_anno[2] = obj['obj_bb'][3] + obj['obj_bb'][1]
                bndbox_anno[3] = obj['obj_bb'][2] + obj['obj_bb'][0]
                bbox.append(bndbox_anno)
                pose.append(np.vstack((rot_anno, trans_anno)))
                name = obj['obj_id'] - 1
                assert(name == classId)
                label.append(name)
            #print (bbox, label)        
            bbox = np.stack(bbox).astype(np.float32)
            pose = np.stack(pose).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
                    
            img_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), mode,  str(id_).zfill(4) + '.jpg')
            img = read_image(img_file, color=True)

            return img, bbox, pose, label, difficult
            
                    
        else:
            id_ = i/5 + 1
            blendstyle = SYNDATA_BLEND_TYPES[i%5]
            anno = ET.parse(
                os.path.join(self.data_dir, 'annotations', str(int(id_)) + '.xml'))
            bbox = list()
            label = list()
            difficult = list()
            for obj in anno.findall('object'):
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                if not self.use_difficult and int(obj.find('difficult').text) == 1:
                    continue

                difficult.append(int(obj.find('difficult').text))
                bndbox_anno = obj.find('bndbox')
                # subtract 1 to make pixel indexes 0-based
                bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
                name = int(obj.find('name').text.lower().strip()) - 1
                label.append(name)
            #print (bbox, label)        
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
                    
            img_file = os.path.join(self.data_dir, 'images', str(int(id_)) + '_' + blendstyle + '.jpg')
            img = read_image(img_file, color=True)

            return img, bbox, label, difficult

    __getitem__ = get_example


TEJANI_BBOX_LABEL_NAMES = (
    'camera',
    'cup',
    'joystick',
    'juice',
    'milk',
    'shampoo')

SYNDATA_BLEND_TYPES = ['box', 'gaussian', 'motion', 'none', 'poisson']
SEQ_COUNTS = [265, 414, 543, 410, 95, 340]
TEST_COUNTS = []
TRAINVAL_COUNTS = []
TRAINVAL_IDS = {}
TEST_IDS = {}

for i in range(len(SEQ_COUNTS)):
    np.random.seed(0)
    scrambled_ids = np.random.permutation(SEQ_COUNTS[i])
    boundary = round(0.3*SEQ_COUNTS[i])
    TRAINVAL_COUNTS += [boundary]
    TEST_COUNTS += [SEQ_COUNTS[i] - boundary]
    TRAINVAL_IDS[i] = scrambled_ids[0:boundary]
    TEST_IDS[i] = scrambled_ids[boundary:SEQ_COUNTS[i]]

#print(TEST_IDS[0])
#print(TRAINVAL_IDS[0])

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

def recover_6d_pose(pose_vec, bbox):
    # implements the rodriguez exponential map to recover the 3x3 rotation matrix and 
    # calculates the translation vector using the regressed depth and bounding box
    # inputs: pose_vec - a 3x1 vector in so(3)
    # bbox: a 4x1 vector in the form of (ymin, xmin, ymax, xmax)
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

    return rot_mat, t_vec

# dummyTD = TejaniBboxDataset('/home/pufik/fyp/tejani_et_al/test/', split='test')

# for i in range(1):
#     #     for j in range(4):
#     #         if math.isnan(example[j]):
#     #             print("NaN found in example ", i)
#     print("Example  ", i, " :", dummyTD.get_example(i)[2])
#     ex = dummyTD.get_example(i)[2][0]
#     posevec = transform_pose_mat(ex)
#     print("posevec: ", posevec)
#     pose = recover_6d_pose(posevec, dummyTD.get_example(i)[1][0])
#     print("Recovered pose:  ", i, " : \n", pose)


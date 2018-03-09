import os
import xml.etree.ElementTree as ET
import yaml
import time

import numpy as np

from .util import read_image


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
        else:
            id_list_file = os.path.join(
                data_dir, 'train.txt')

            self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = TEJANI_BBOX_LABEL_NAMES
        self.split = split

    def __len__(self):
        print(len(self.ids))
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        if self.split == 'test':
            id_ = 0
            classId = 0
            while True:
                if (i < sum(TEST_COUNTS[:classId + 1]) and i >= sum(TEST_COUNTS[:classId])):
                    id_ = i - sum(TEST_COUNTS[:classId])
                    #print("Class id: ", classId)
                    break
                classId += 1                 
            bbox = list()
            label = list()
            difficult = list()
            for obj in self.anno[classId][id_]:
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                bndbox_anno = obj['obj_bb']
                difficult.append(0)
                # subtract 1 to make pixel indexes 0-based
                bndbox_anno[0] = obj['obj_bb'][1]
                bndbox_anno[1] = obj['obj_bb'][0]
                bndbox_anno[2] = obj['obj_bb'][3] + obj['obj_bb'][1]
                bndbox_anno[3] = obj['obj_bb'][2] + obj['obj_bb'][0]
                bbox.append(bndbox_anno)
                name = obj['obj_id'] - 1
                assert(name == classId)
                label.append(name)
            #print (bbox, label)        
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
                    
            img_file = os.path.join(self.data_dir, str(classId + 1).zfill(2), 'rgb',  str(id_).zfill(4) + '.jpg')
            img = read_image(img_file, color=True)

            return img, bbox, label, difficult
            
                    
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
TEST_COUNTS = [265, 414, 543, 410, 95, 340]

#dummyTD = TejaniBboxDataset('/home/pufik/fyp/syndata-generation/myoutput/', split='trainval')

# for i in range(0, 1):
#     print("Example  ", i, " :", dummyTD.get_example(i))


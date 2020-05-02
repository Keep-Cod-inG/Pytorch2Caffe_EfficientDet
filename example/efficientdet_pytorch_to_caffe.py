# Author: Zyz

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
from torch.autograd import Variable

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
import pytorch_to_caffe
compound_coef = 0
force_input_size = None  # set None to use default size
img_path = '/home/tt/projects/PytorchToCaffe/test/img.png'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = False
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
# ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
#
# if use_cuda:
#     x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
# else:
#     x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
#
# x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

input=Variable(torch.ones([1,3,512,512]))

net = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

net.load_state_dict(torch.load('./efficientdet-d0.pth'))
net.requires_grad_(False)
net.eval()
name='efficientdet'
pytorch_to_caffe.trans_net(net,input,name)
pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
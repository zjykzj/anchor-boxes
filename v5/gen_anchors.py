# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:01
@File    : gen_anchors.py
@Author  : zj
@Description: 
"""

import os
import cv2
import glob
import torch

from tqdm import tqdm
import numpy as np

from autoanchor import check_anchor_order, check_anchors, kmean_anchors


class Model:

    def __init__(self, anchors=None, stride=None):
        """

        :param anchors: [Nl, Na, 2], where Nl is the number of detection layers, Na is the number of anchors, and 2 is the width and height of each anchor
        :param stride: [Nl], where Nl is the scaling factor of each detection layer relative to the input image
        """
        self.anchors = torch.from_numpy(np.array(anchors, dtype=float)).view(len(anchors), -1, 2)
        self.stride = torch.from_numpy(np.array(stride, dtype=int))
        assert len(self.anchors) == len(self.stride)


class Dataset:

    def __init__(self, shapes=None, labels=None):
        """
        :param shapes: [N, 2], where N is the number of images and 2 is the width and height of each image
        :param labels: [N, ...], where N is the number of images, each item in the list contains the annotation box width and height of the corresponding image
        """
        self.shapes = shapes
        self.labels = labels
        assert len(self.shapes) == len(self.labels)


def test_model():
    anchors = np.array([
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ])
    stride = [8, 16, 32]
    m = Model(anchors=anchors, stride=stride)

    print("anchors:", m.anchors)
    print("stride:", m.stride)
    check_anchor_order(m)
    print("anchors:", m.anchors)
    print("stride:", m.stride)


def get_yolov5_data(root, name):
    image_dir = os.path.join(root, name, "images")
    assert os.path.isdir(image_dir), image_dir
    label_dir = os.path.join(root, name, 'labels')
    assert os.path.isdir(label_dir), label_dir
    label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

    shapes = list()
    labels = list()

    for label_path in tqdm(label_path_list):
        # [[box_w, box_h], ]
        # The coordinate size is relative to the width and height of the image
        boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
        if len(boxes) == 0:
            continue
        if len(boxes.shape) == 1:
            boxes = [boxes]
        # for label, xc, yc, box_w, box_h in boxes:
        #     box_list.append([box_w, box_h])
        labels.append(np.array(boxes))

        image_name = os.path.basename(label_path).replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        assert os.path.isfile(image_path), image_path
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        shapes.append((w, h))

    return np.array(shapes), labels


def test_dataset():
    shapes, labels = get_yolov5_data("../../datasets/", "voc2yolov5-train")
    dataset = Dataset(shapes, labels)

    anchors = np.array([
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ])
    stride = [8, 16, 32]
    m = Model(anchors=anchors, stride=stride)

    check_anchors(dataset, m)
    print('anchors:', m.anchors)


if __name__ == '__main__':
    test_dataset()
    # test_model()

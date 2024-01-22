# -*- coding: utf-8 -*-

"""
@date: 2024/1/21 下午7:52
@file: draw_anchors.py
@author: zj
@description:
"""

import os
import cv2
import numpy as np

STRIDE = 32


def load_anchors(txt_path, size=1, stride=32):
    assert os.path.isfile(txt_path), txt_path

    with open(txt_path, 'r') as f:
        anchors = f.readline().strip().split(",")
        anchors = np.array(anchors, dtype=float).reshape(-1, 2) * size * stride
    # print(anchors)

    # indices = np.argsort(anchors[:, 0] * anchors[:, 1])
    # indices = np.argsort(anchors.prod(-1))
    # indices = anchors.prod(-1).argsort()
    # anchors = anchors[indices]
    anchors = anchors[anchors.prod(-1   ).argsort()]

    return anchors


def draw_darknet():
    txt_path = "darknet/generated_anchors/anchors/anchors5.txt"
    anchors = load_anchors(txt_path, stride=STRIDE, size=1)

    canva_w = int(np.sum(anchors[:, 0]) + (len(anchors) + 1) * 10)
    canva_h = int(np.max(anchors[:, 1]) + 10 * 2)

    canva = np.ones((canva_h, canva_w, 3), dtype=np.uint8) * 255
    for i, anchor in enumerate(anchors):
        center_h = canva_h / 2
        center_w = (i + 1) * 10 + np.sum(anchors[:i, 0]) + anchor[0] / 2.

        xmin = center_w - anchor[0] / 2.
        ymin = center_h - anchor[1] / 2.
        xmax = center_w + anchor[0] / 2.
        ymax = center_h + anchor[1] / 2.
        cv2.rectangle(canva, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 1)

    # cv2.imshow("canva", canva)
    # cv2.waitKey()
    return canva


def draw_anchors(voc_path, coco_path, canva_name="voc_coco"):
    anchors_voc = load_anchors(voc_path, stride=STRIDE, size=13)
    anchors_coco = load_anchors(coco_path, stride=STRIDE, size=13)

    final_canva_w = 0
    final_canvs_h = 0

    canva_list = list()
    for i, (a_voc, a_coco) in enumerate(zip(anchors_voc, anchors_coco)):
        canva_w = np.max((a_voc[0], a_coco[0])) + 4
        canva_h = np.max((a_voc[1], a_coco[1])) + 4
        canva = np.ones((int(canva_h), int(canva_w), 3), dtype=np.uint8) * 255

        xmin = (canva_w - a_voc[0]) / 2.
        ymin = (canva_h - a_voc[1]) / 2.
        xmax = (canva_w + a_voc[0]) / 2.
        ymax = (canva_h + a_voc[1]) / 2.
        cv2.rectangle(canva, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 1)

        xmin = (canva_w - a_coco[0]) / 2.
        ymin = (canva_h - a_coco[1]) / 2.
        xmax = (canva_w + a_coco[0]) / 2.
        ymax = (canva_h + a_coco[1]) / 2.
        cv2.rectangle(canva, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (200, 200, 200), -1)

        canva_list.append(canva)
        final_canva_w += canva_w
        final_canvs_h = max(final_canvs_h, canva_h)

        # cv2.imshow("canva", canva)
        # cv2.waitKey(0)

    final_canva_w += 10 * (len(canva_list) + 1)
    final_canvs_h += 10 * 2
    final_canva = np.ones((int(final_canvs_h), int(final_canva_w), 3), dtype=np.uint8) * 255

    center_h = final_canvs_h / 2
    center_w = 0
    for canva in canva_list:
        center_w += 10 + canva.shape[1] / 2.

        xmin = int(center_w - canva.shape[1] / 2.)
        ymin = int(center_h - canva.shape[0] / 2.)
        xmax = int(center_w + canva.shape[1] / 2.)
        ymax = int(center_h + canva.shape[0] / 2.)
        final_canva[ymin:ymax, xmin:xmax] = canva

        center_w += canva.shape[1] / 2.

    # cv2.imshow(canva_name, final_canva)
    # cv2.waitKey()
    return final_canva


if __name__ == '__main__':
    canva_darknet = draw_darknet()
    cv2.imwrite("assets/canva_darknet.jpg", canva_darknet)

    voc_path = "v2/generated_anchors/voc/anchors5.txt"
    coco_path = "v2/generated_anchors/coco/anchors5.txt"
    canva_v2 = draw_anchors(voc_path, coco_path, canva_name="v2")
    cv2.imwrite("assets/canva_v2.jpg", canva_v2)

    voc_path = "v3/generated_anchors/voc/anchors5.txt"
    coco_path = "v3/generated_anchors/coco/anchors5.txt"
    canva_v3 = draw_anchors(voc_path, coco_path, canva_name="v3")
    cv2.imwrite("assets/canva_v3.jpg", canva_v3)

# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 15:01
@File    : gen_anchors.py
@Author  : zj
@Description: 
"""

import numpy as np
import torch

from autoanchor import check_anchor_order


class Model:

    def __init__(self, anchors=None, stride=None):
        self.anchors = torch.from_numpy(np.array(anchors, dtype=float))
        self.stride = stride


if __name__ == '__main__':
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

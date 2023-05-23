# -*- coding: utf-8 -*-

"""
@date: 2023/5/1 下午9:13
@file: gen_anchors.py
@author: zj
@description:
"""

import os
import glob
import random
import argparse

import numpy as np

import warnings

warnings.filterwarnings('ignore')

width_in_cfg_file = 416.
height_in_cfg_file = 416.

strides = 32.


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv2 Anchor-boxes.")
    parser.add_argument('data', metavar='DIR', help='Path to dataset')
    parser.add_argument('-t', '--train', metavar='TRAIN', default='voc2yolov5-train', help='Train dataset')
    parser.add_argument('-v', '--val', metavar='VAL', default='voc2yolov5-val', help='Val dataset')
    parser.add_argument('output', metavar='OUTPUT', help='Path to save files')
    parser.add_argument('-e', '--exp', metavar='EXP', default='voc', help='Sub-folder name')

    parser.add_argument('-n', '--num-clusters', metavar='NUM', default=None,
                        help='Number of cluster centroids')

    args = parser.parse_args()
    print("args:", args)

    return args


def get_yolov5_data(root, name):
    label_dir = os.path.join(root, name, 'labels')
    label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

    box_list = list()
    for label_path in label_path_list:
        # [[box_w, box_h], ]
        # The coordinate size is relative to the width and height of the image
        boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
        if len(boxes) == 0:
            continue
        if len(boxes.shape) == 1:
            boxes = [boxes]
        for label, xc, yc, box_w, box_h in boxes:
            box_list.append([box_w, box_h])
    box_array = np.array(box_list, dtype=float)
    return box_array


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def init_centroids(box_array, K):
    # See https://zhuanlan.zhihu.com/p/32375430
    N = len(box_array)
    c0_idx = int(np.random.uniform(0, N))
    centroids = box_array[c0_idx].reshape(1, -1)  # 选择第一个簇中心

    k = 1
    while k < K:
        d2 = []
        for i in range(N):
            d = 1 - IOU(box_array[i], centroids)
            d2.append(np.min(d))
            # subs = centroid - x[i, :]
            # dimension2 = np.power(subs, 2)
            # dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            # d2.append(np.min(dimension_s))

        # ---- 直接选择概率值最大的 ------
        # new_c_idx = np.argmax(d2)
        # ---- 依照概率分布进行选择 -----
        prob = np.array(d2) / np.sum(np.array(d2))
        new_c_idx = np.random.choice(N, p=prob)

        centroids = np.vstack([centroids, box_array[new_c_idx]])
        k += 1
    return centroids


def kmeans(box_array, centroids, eps):
    N = box_array.shape[0]
    K, dim = centroids.shape

    prev_assignments = np.ones(N) * (-1)
    old_D = np.zeros((N, K))

    iter = 0
    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(box_array[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            break

        # calculate new centroids
        centroid_sums = np.zeros((K, dim), dtype=float)
        for i in range(N):
            centroid_sums[assignments[i]] += box_array[i]
        for j in range(K):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

    return centroids


def write_anchors_to_file(centroids, train_box_array, test_box_array, anchor_file):
    anchors = centroids.copy()
    scaled_anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        scaled_anchors[i][0] *= width_in_cfg_file / strides
        scaled_anchors[i][1] *= height_in_cfg_file / strides

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)
    print('Anchors = ', anchors[sorted_indices])
    print('Scaled Anchors = ', scaled_anchors[sorted_indices])

    train_avg_iou = avg_IOU(train_box_array, centroids)
    test_avg_iou = avg_IOU(test_box_array, centroids)
    print(f'Train Avg IOU: {train_avg_iou}')
    print(f'Test Avg IOU: {test_avg_iou}')

    print(f"Write to {anchor_file}")
    with open(anchor_file, 'w') as f:
        anchors = [str('%.2f' % x) for x in anchors[sorted_indices].reshape(-1)]
        scaled_anchors = [str('%.2f' % x) for x in scaled_anchors[sorted_indices].reshape(-1)]

        f.write(','.join(anchors) + '\n')
        f.write(','.join(scaled_anchors) + '\n')

        f.write('%f\n' % (train_avg_iou))
        f.write('%f\n' % (test_avg_iou))
        print()


def process(num_clusters, train_box_array, test_box_array, anchor_file):
    eps = 0.005

    # indices = [random.randrange(train_box_array.shape[0]) for _ in range(num_clusters)]
    # centroids = train_box_array[indices]
    centroids = init_centroids(train_box_array, num_clusters)
    print("Init centroids:\n", centroids)

    centroids = kmeans(train_box_array, centroids, eps)
    write_anchors_to_file(centroids, train_box_array, test_box_array, anchor_file)


def main():
    args = parse_args()

    train_box_array = get_yolov5_data(args.data, args.train)
    test_box_array = get_yolov5_data(args.data, args.val)

    output_dir = os.path.join(args.output, args.exp)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if args.num_clusters is None:
        for ni in range(1, 11):
            num_clusters = ni
            anchor_file = os.path.join(output_dir, f'anchors{num_clusters}.txt')
            process(num_clusters, train_box_array, test_box_array, anchor_file)
    else:
        num_clusters = int(args.num_clusters)
        anchor_file = os.path.join(output_dir, f'anchors{num_clusters}.txt')
        process(num_clusters, train_box_array, test_box_array, anchor_file)


if __name__ == '__main__':
    main()

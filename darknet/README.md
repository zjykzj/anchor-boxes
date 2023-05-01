
# Darknet

## YOLOv2

[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/tree/master) provides python implementation for YOLOv2 network anchor-boxes.

* Using [voc_label.py](scripts/voc_label.py) to get `train.txt`
* Using [gen_anchors.py](scripts/gen_anchors.py) to compute anchor-boxes

**Notes:**

1. From the implementation, it can be observed that the training set includes voc2007-trainval + voc2012-trainval
2. There are always slight differences in the calculation results of each anchor-box
3. Darknet also provides YOLOv2 training recipes, but there will be significant differences with mine simulation anchor-boxes. Didn't find a specific reason?
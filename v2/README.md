
# YOLOv2

## Pascal VOC

### Prepare data

See [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py) to get the train/test dataset

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

### Generate anchor-boxes

Generate a specified number of anchor-boxes lists

```
python gen_anchors.py -t voc2yolov5-train -v voc2yolov5-val -n 5 -e voc /home/zj/data/voc ./generated_anchors
```

Or traverse anchor-boxes with different numbers [1, 10]. See [generated_anchors/voc](generated_anchors/voc)

```
python gen_anchors.py -t voc2yolov5-train -v voc2yolov5-val -e voc /home/zj/data/voc ./generated_anchors
```

The content of `anchors5.txt` is as follows:

```text
# Anchors
0.09,0.15,0.21,0.35,0.35,0.69,0.62,0.41,0.79,0.82
# Scaled Anchors (Anchors * 13.)
1.19,1.99,2.79,4.60,4.53,8.92,8.06,5.29,10.32,10.65
# Train Avg IOU
0.615898
# Test Avg IOU
0.622052
```

## COCO

### Prepare data

See [coco2yolov5.py](https://github.com/zjykzj/cocodev/blob/master/py/coco2yolov5.py) to get the train/test dataset

```shell
python coco2yolov5.py /home/zj/data/coco ./coco2yolov5-train --name train2017
python coco2yolov5.py /home/zj/data/coco ./coco2yolov5-val --name val2017
```

### Generate anchor-boxes

Generate a specified number of anchor-boxes lists

```
python gen_anchors.py -t coco2yolov5-train -v coco2yolov5-val -n 5 -e coco /home/zj/data/coco ./generated_anchors
```

Or traverse anchor-boxes with different numbers [1, 10]. See [generated_anchors/coco](generated_anchors/coco)

```
python gen_anchors.py -t coco2yolov5-train -v coco2yolov5-val -e coco /home/zj/data/coco ./generated_anchors
```

The content of `anchors5.txt` is as follows:

```text
# Anchors
0.04,0.06,0.13,0.18,0.22,0.50,0.49,0.29,0.69,0.75
# Scaled Anchors (Anchors * 13.)
0.53,0.80,1.71,2.36,2.90,6.45,6.34,3.79,9.03,9.75
# Train Avg IOU
0.494198
# Test Avg IOU
0.496213
```
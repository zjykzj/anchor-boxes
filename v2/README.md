
# YOLOv2

## Pascal VOC

### Prepare data

See [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py) to get the train dataset

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
```

### Generate anchor-boxes

Generate a specified number of anchor box lists

```
python gen_anchors.py -n 5 -e voc /home/zj/data/voc/voc2yolov5-train/ ./generated_anchors
```

Or traverse anchor boxes with different numbers [1, 10]. See [generated_anchors/voc](generated_anchors/voc)

```
python gen_anchors.py -e voc /home/zj/data/voc/voc2yolov5-train/ ./generated_anchors
```
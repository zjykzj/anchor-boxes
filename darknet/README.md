# Darknet

## YOLOv2

[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/tree/master) provides python implementation for YOLOv2 network
anchor-boxes.

* Using [voc_label.py](scripts/voc_label.py) to get `train.txt`
* Using [gen_anchors.py](scripts/gen_anchors.py) to compute anchor-boxes

**Notes:**

1. From the implementation, it can be observed that the training set includes voc2007-trainval + voc2012-trainval
2. There are always slight differences in the calculation results of each anchor-box
3. Darknet also provides YOLOv2 training recipes, but there will be significant differences with mine simulation
   anchor-boxes. Didn't find a specific reason?

For example, you can see anchors in

* [darknet/cfg/yolov2.cfg](https://github.com/AlexeyAB/darknet/blob/ed59050950b5a890a2a1d1c69547250c436a5968/cfg/yolov2.cfg#L242)

   ```
   anchors =  0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
   ```

* [darknet/cfg/yolov2-voc.cfg](https://github.com/AlexeyAB/darknet/blob/ed59050950b5a890a2a1d1c69547250c436a5968/cfg/yolov2-voc.cfg#L242)

   ```
   anchors =  1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071
   ```

* [darknet/cfg/yolov2-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/ed59050950b5a890a2a1d1c69547250c436a5968/cfg/yolov2-tiny.cfg#L123)

   ```
   anchors =  0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
   ```

* [darknet/cfg/yolov2-tiny-voc.cfg](https://github.com/AlexeyAB/darknet/blob/ed59050950b5a890a2a1d1c69547250c436a5968/cfg/yolov2-tiny-voc.cfg#L122)

  ```
  anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
  ```

Mine simulation results in [generated_anchors/anchors](generated_anchors/anchors)

# yolov3-detector
Fast YOLOv3 detector using PyTorch backend.

* This library is useful for implementing YOLO detector with the configuration and the weights file. See `main.py` for basic usage.
* Training is not possible with this library. The best way to train the network is to use the original [pjreddie/darknet](https://github.com/pjreddie/darknet) or [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) which is a slightly improved version of the original one. I strongly recommend Alexey's version since documentation is perfect and the code is actively being developed. There is also full PyTorch implementation [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) that enables training.  
* For more information see the [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

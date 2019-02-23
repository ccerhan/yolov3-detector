# yolov3-detector
Fast YOLOv3 detector using PyTorch backend.

* This library is useful for implementing YOLO detector with the configuration and the weights file. See `main.py` for basic usage.
* Training is not possible with this library. The best way to train the network is to use the original [pjreddie/darknet](https://github.com/pjreddie/darknet) or [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) which is a slightly improved version of the original one. I strongly recommend Alexey's version since documentation is perfect and the code is actively being developed. 
* Some of the code such as parsing configuration and weight files are taken from [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) which is a full PyTorch implementation that also enables training. In addition, several bugs are solved and YOLO layer is improved to be able to detect image sizes different than square sizes (see `darknet.py`). 
* For more information see the [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

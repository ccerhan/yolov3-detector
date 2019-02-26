# yolov3-detector
Fast YOLOv3 detector using PyTorch backend.

* This library is useful for implementing YOLO detector with the configuration and the weights file. See `main.py` for basic usage. 
* Training is not possible with this library. The best way to train the network is to use the original [pjreddie/darknet](https://github.com/pjreddie/darknet) or [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) which is a slightly improved version of the original one. I strongly recommend Alexey's version since documentation is perfect and the code is actively being developed.  
* Some of the code such as parsing configuration and weight files are taken from [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) which is a full PyTorch implementation of darknet that also enables training but not detection. 
* YOLO layer is improved to be able to detect image sizes different than square sizes such as `416x608` (see `darknet.py`). Note that width and height must be multiples of 32. This feature enables less computation for images with wide aspect ratio since it prevents unnecessary computation on paddings resulting from letterbox resizing.
* For more information see the [original paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

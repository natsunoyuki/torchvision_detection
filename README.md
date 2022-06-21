# TORCHVISION DETECTION
`torchvision_detection` is a package of utility functions for performing object detection using torchvision models. The functions included in this package are mainly compiled from several sources into one common pool for easy access.

### Torchvision Training References
These functions and scripts are not part of the `torch` package, and are originally obtained from:
https://github.com/pytorch/vision/tree/main/references/detection

### Modern Computer Vision with PyTorch Scripts
These functions are originally from the GitHub repository of Modern Computer Vision with PyTorch by V Kishore Ayyadevara and Yeshwanth Reddy:
https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch

# Installation
```
pip install git+https://github.com/natsunoyuki/torchvision_detection
```

# Usage
```
import detection

dataset = detection.coco_utils.get_coco(root, image_set, transform, mode = "instances")
```

# References
https://pytorch.org/vision/stable/training_references.html

https://github.com/pytorch/vision/tree/main/references/detection

https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter08/Training_Faster_RCNN.ipynb
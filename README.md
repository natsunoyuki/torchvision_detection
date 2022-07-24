# TORCHVISION DETECTION
`torchvision_detection` is a package of utility functions for performing object detection using torchvision models. The functions included in this package are mainly compiled from several sources into one common repository for easy access and use. These functions were aggregated directly from the sources below.

### Torchvision Training References
These functions and scripts are not part of the `torch` package, and were obtained from:
https://github.com/pytorch/vision/tree/v0.12.0/references/detection

### Modern Computer Vision with PyTorch Scripts
These functions were obtained from the GitHub repository of Modern Computer Vision with PyTorch by V Kishore Ayyadevara and Yeshwanth Reddy:
https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter08/Training_Faster_RCNN.ipynb

# Installation
```
pip install git+https://github.com/natsunoyuki/torchvision_detection
```

# Usage
```
import detection
import fasterrcnn_utils

# Coco dataset.
dataset = detection.coco_utils.get_coco(root, image_set, transform, mode = "instances")

# Fasterrcnn model.
model = fasterrcnn_utils.model.fasterrcnn(num_classes = 2)
```

# References

* https://pytorch.org/vision/stable/training_references.html
* https://github.com/pytorch/vision/tree/v0.12.0/references/detection
* https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch
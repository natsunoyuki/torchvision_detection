from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn


################################################################################
# These are helper functions used to download the faster rcnn object detection
# model and the mask rcnn segmentation model from torchvision.
# Original Python code from:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
################################################################################

def fasterrcnn(num_classes = 2, feature_extraction = True):
    model = fasterrcnn_resnet50_fpn(pretrained = True)

    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def maskrcnn(num_classes = 2, feature_extraction = True):
    model = maskrcnn_resnet50_fpn(pretrained = True)

    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


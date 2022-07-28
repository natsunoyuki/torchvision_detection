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
    """Downloads the torchvision fasterrcnn model.
    https://pytorch.org/vision/0.12/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

    Arguments
        num_classes: int
            Number of classes to predict. Note that the background (index = 0) counts as a class!
            Therefore if we are predicting cats and dogs we have a total of 3 classes!
        feature_extraction: bool
            Use the pretrained weights for feature extraction. If set to true,
            only the classification layer weights will be trainable. If set to
            false, all weights in the model well be set to trainable.
    
    Returns
        model: FasterRCNN
            Torch pretrained fasterrcnn object detection model.
    """
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


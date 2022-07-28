import numpy as np
import torch
import torchvision
#import matplotlib.pyplot as plt
from torch_snippets import Report


################################################################################
# These are helper functions used to train torchvision object detection models.
# Original Python code by V Kishore Ayyadevara and Yeshwanth Reddy:
# https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/
################################################################################

OUTPUT_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]

def unbatch(batch, device):
    X, y = batch
    # Move data to the specified device.
    X = [i.to(device) for i in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y

def get_reduction_function(reduction = "sum"):
    if reduction == "sum":
        return np.sum
    elif reduction == "mean":
        return np.mean

def train_batch(batch, model, optimizer, device, reduction = "sum"):
    reduction = get_reduction_function(reduction)

    model.train()

    X, y = unbatch(batch, device = device)

    # Forward pass.
    optimizer.zero_grad()
    losses = model(X, y)
    loss = reduction(loss for loss in losses.values())

    # Back-propagation of gradients.
    loss.backward()
    optimizer.step()

    return loss, losses

@torch.no_grad()
def validate_batch(batch, model, optimizer, device, reduction = "sum"):
    reduction = get_reduction_function(reduction)

    # For training validation, we set the model to train as we want the model
    # to return us the losses instead of the predictions.
    model.train() 

    X, y = unbatch(batch, device = device)

    # Forward pass to get the losses only.
    optimizer.zero_grad()
    losses = model(X, y)
    loss = reduction(loss for loss in losses.values())

    return loss, losses

def train_fasterrcnn(model, optimizer, n_epochs, train_loader, val_loader = None, log = None, device = "cpu"):
    if log is None:
        log = Report(n_epochs)

    # Move the model to the specified device. Both the model and the data must be on the same device!
    model.to(device)

    for epoch in range(n_epochs):
        # Train pass.
        n_batch = len(train_loader) # Number of training batches.
        for i, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model, optimizer, device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
            pos = epoch + (i + 1) / n_batch
            log.record(pos = pos, trn_loss = loss.item(), trn_loc_loss = loc_loss.item(), 
                       trn_regr_loss = regr_loss.item(), trn_objectness_loss = loss_objectness.item(),
                       trn_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = "\r")

        if val_loader is not None:
            # Validation pass.
            n_batch = len(val_loader) # Number of validation batches.
            for i, batch in enumerate(val_loader):
                loss, losses = validate_batch(batch, model, optimizer, device)
                loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
                pos = epoch + (i + 1) / n_batch
                log.record(pos = pos, val_loss = loss.item(), val_loc_loss = loc_loss.item(), 
                           val_regr_loss = regr_loss.item(), val_objectness_loss = loss_objectness.item(),
                           val_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = "\r")

    log.report_avgs(epoch + 1)

    return log

@torch.no_grad()
def predict_batch(batch, model, device):
    model.to(device)
    # For predictions, we set the model to eval in order
    # to get the bounding box and classification predictions
    # instead of the losses.
    model.eval()
    X, _ = unbatch(batch, device = device)
    predictions = model(X)
    return predictions

def predict(model, data_loader, device = "cpu"):
    predictions = []
    for i, batch in enumerate(data_loader):
        predictions = predictions + predict_batch(batch, model, device)
    return predictions

def decode_prediction(prediction, score_threshold = 0.8, nms_iou_threshold = 0.2):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]

    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(boxes = boxes, scores = scores, iou_threshold = nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]

    return boxes, labels, scores
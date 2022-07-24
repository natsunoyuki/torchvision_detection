import numpy as np
import torch
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

    model.to(device)
    model.train()
    X, y = unbatch(batch)

    optimizer.zero_grad()
    losses = model(X, y)
    loss = reduction(loss for loss in losses.values())

    loss.backward()
    optimizer.step()

    return loss, losses

@torch.no_grad()
def validate_batch(batch, model, optimizer, device, reduction = "sum"):
    reduction = get_reduction_function(reduction)

    model.to(device)
    model.train() 
    X, y = unbatch(batch)

    optimizer.zero_grad()
    losses = model(X, y)
    loss = reduction(loss for loss in losses.values())

    return loss, losses

def train_fasterrcnn(model, optimizer, n_epochs, train_loader, test_loader = None, log = None, device = "cpu"):
    if log is None:
        log = Report(n_epochs)

    for epoch in range(n_epochs):
        # Train pass.
        n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, losses = train_batch(inputs, model, optimizer, device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
            pos = (epoch + (ix + 1) / n)
            log.record(pos = pos, trn_loss = loss.item(), trn_loc_loss = loc_loss.item(), 
                       trn_regr_loss = regr_loss.item(), trn_objectness_loss = loss_objectness.item(),
                       trn_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = "\r")

        if test_loader is not None:
            # Test pass.
            n = len(test_loader)
            for ix,inputs in enumerate(test_loader):
                loss, losses = validate_batch(inputs, model, optimizer, device)
                loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
                pos = (epoch + (ix + 1) / n)
                log.record(pos = pos, val_loss = loss.item(), val_loc_loss = loc_loss.item(), 
                           val_regr_loss = regr_loss.item(), val_objectness_loss = loss_objectness.item(),
                           val_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = "\r")

    return log

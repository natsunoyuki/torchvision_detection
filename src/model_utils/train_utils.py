# https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter08/Training_Faster_RCNN.ipynb

import torch

# https://pypi.org/project/torch-snippets/
from torch_snippets import Report


OUTPUT_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def train_batch(inputs, model, optimizer, device):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()

    return loss, losses

@torch.no_grad()
def validate_batch(inputs, model, optimizer, device):
    model.train() 
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())

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
            log.record(pos, trn_loss = loss.item(), trn_loc_loss = loc_loss.item(), 
                       trn_regr_loss = regr_loss.item(), trn_objectness_loss = loss_objectness.item(),
                       trn_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = '\r')

        if test_loader is not None:
            # Test pass.
            n = len(test_loader)
            for ix,inputs in enumerate(test_loader):
                loss, losses = validate_batch(inputs, model, optimizer, device)
                loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
                pos = (epoch + (ix + 1) / n)
                log.record(pos, val_loss = loss.item(), val_loc_loss = loc_loss.item(), 
                           val_regr_loss = regr_loss.item(), val_objectness_loss = loss_objectness.item(),
                           val_rpn_box_reg_loss = loss_rpn_box_reg.item(), end = '\r')

        if (epoch + 1) % (n_epochs // 5) == 0: 
            log.report_avgs(epoch + 1)

    return log
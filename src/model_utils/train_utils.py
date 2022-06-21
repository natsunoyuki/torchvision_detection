# Original code from:
# https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/

import torch
import matplotlib.pyplot as plt


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

class Report(object):
    def __init__(self):
        self.trn_pos = []
        self.trn_loss = []
        self.trn_loc_loss = []
        self.trn_regr_loss = [] 
        self.trn_objectness_loss = []
        self.trn_rpn_box_reg_loss = []
    
        self.val_pos = []
        self.val_loss = []
        self.val_loc_loss = []
        self.val_regr_loss = []
        self.val_objectness_loss = []
        self.val_rpn_box_reg_loss = []

    def record(self, 
               trn_pos = None, 
               trn_loss = None, 
               trn_loc_loss = None, 
               trn_regr_loss = None, 
               trn_objectness_loss = None,
               trn_rpn_box_reg_loss = None,
               val_pos = None,
               val_loss = None, 
               val_loc_loss = None, 
               val_regr_loss = None, 
               val_objectness_loss = None,
               val_rpn_box_reg_loss = None):
        
        if trn_pos is not None:
            self.trn_pos.append(trn_pos)

        if trn_loss is not None:
            self.trn_loss.append(trn_loss)
        
        if trn_loc_loss is not None:
            self.trn_loc_loss.append(trn_loc_loss)

        if trn_regr_loss is not None:
            self.trn_regr_loss.append(trn_regr_loss)

        if trn_objectness_loss is not None:
            self.trn_objectness_loss.append(trn_objectness_loss)

        if trn_rpn_box_reg_loss is not None:
            self.trn_rpn_box_reg_loss.append(trn_rpn_box_reg_loss)

        if val_pos is not None:
            self.val_pos.append(val_pos)

        if val_loss is not None:
            self.val_loss.append(val_loss)

        if val_loc_loss is not None:
            self.val_loc_loss.append(val_loc_loss)

        if val_regr_loss is not None:
            self.val_regr_loss.append(val_regr_loss)

        if val_objectness_loss is not None:
            self.val_objectness_loss.append(val_objectness_loss)

        if val_rpn_box_reg_loss is not None:
            self.val_rpn_box_reg_loss.append(val_rpn_box_reg_loss)

    def plot_epochs(self, vars = ["trn_loss", "val_loss"], figsize = [10, 5]):

        d = {"trn_loss": self.trn_loss, "val_loss": self.val_loss}

        plt.figure(figsize = figsize)
        for v in vars:
            plt.plot(d.get(v, None))

        plt.show()


def train_fasterrcnn(model, optimizer, n_epochs, train_loader, test_loader = None, log = None, device = "cpu"):
    if log is None:
        log = Report()

    for epoch in range(n_epochs):
        # Train pass.
        n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, losses = train_batch(inputs, model, optimizer, device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
            pos = (epoch + (ix + 1) / n)
            log.record(trn_pos = pos, trn_loss = loss.item(), trn_loc_loss = loc_loss.item(), 
                       trn_regr_loss = regr_loss.item(), trn_objectness_loss = loss_objectness.item(),
                       trn_rpn_box_reg_loss = loss_rpn_box_reg.item())

        if test_loader is not None:
            # Test pass.
            n = len(test_loader)
            for ix,inputs in enumerate(test_loader):
                loss, losses = validate_batch(inputs, model, optimizer, device)
                loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in OUTPUT_KEYS]
                pos = (epoch + (ix + 1) / n)
                log.record(val_pos = pos, val_loss = loss.item(), val_loc_loss = loc_loss.item(), 
                           val_regr_loss = regr_loss.item(), val_objectness_loss = loss_objectness.item(),
                           val_rpn_box_reg_loss = loss_rpn_box_reg.item())

        #if (epoch + 1) % (n_epochs // 5) == 0: 
        #    log.report_avgs(epoch + 1)

    return log
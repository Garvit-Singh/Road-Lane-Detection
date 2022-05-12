import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)
        
        BCE =  nn.BCEWithLogitsLoss()
        BCE = BCE(inputs, targets)
        intersection = torch.sum(torch.dot(targets, inputs))    
        dice_loss = 1 - (2*intersection + smooth) / (torch.sum(targets) + torch.sum(inputs) + smooth)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
       
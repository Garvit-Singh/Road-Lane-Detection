import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, epsilon = 1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, pred, target):
        # batch_size x 1 x 96 x 192
        loss = -target*(torch.log(pred + self.epsilon)*(1-pred)**self.gamma) - (1-target)*(torch.log((1-pred) + self.epsilon)*(pred)**self.gamma)
        return torch.mean(loss)
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, targets, inputs):
        # batch_size x 1 x 96 x 192

        #flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = torch.sum(targets * inputs)
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))  # wrong lanes 
        
        IoULoss = (TP + self.epsilon) / (TP + self.alpha*FP + self.beta*FN + self.epsilon) 
        return 1-IoULoss
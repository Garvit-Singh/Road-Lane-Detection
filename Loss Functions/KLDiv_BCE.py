from typing_extensions import Self
import numpy as np
import torch
import torch.nn as nn

class KLDiv_BCE(nn.Module):
    def __init__(self, sigma=2, epsilon=1e-6):
        super(KLDiv_BCE, self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon

    def loss_bg(self, inputs, targets):
        # inputs : 1026 x 96 x 192
        # tagets : 1026 x 96 x 192
        bce = nn.BCEWithLogitsLoss(reduction="mean")
        loss = bce(inputs, targets)
        return loss

    def loss_pair(self, inputs, targets, px, py, qx, qy):
        # inputs : tensor (batch_size x 7 x 96 x 192)
        # targets : tensor (batch_size x 7 x 96 x 192) 
        n, _, _, _ = inputs.shape
        zr = torch.zeros(size=(n, 7))

        # Pk = batch_size x 7
        Pi = inputs[:, :, px, py]
        Pj = inputs[:, :, qx, qy]
        # Tells about Rij
        Gi = targets[:, :, px, py]
        Gj = targets[:, :, qx, qy]

        # kld 
        kld = nn.KLDivLoss(reduction="batchmean")

        Dki = kld(Pi, Pj)
        Dkj = kld(Pj, Pi)

        if(torch.equal(Gi, Gj)):
            L1 = Dki + Dkj
            return L1
        else :
            L2 = torch.maximum(self.sigma - Dki, zr) + torch.maximum(self.sigma - Dkj, zr)
            L2 = torch.sum(L2)
            return L2

    def forward(self, inputs, targets):
        # inputs : tensor (batch_size x 7 x 96 x 192)
        # targets : tensor (batch_size x 7 x 96 x 192) 
        n, c, h, w = inputs.shape

        # BACKGROUND LOSS
        bg_loss = self.loss_bg(inputs[:, 0, :, :], targets[:, 0, :, :])

        # KLDivergence LOSS (if more time taken to traing use 100 pixels instead)
        cnt = 0
        kl_loss = 0.0
        # find random 100 pixels 
        px = np.random.randint(192, size=(100))
        py = np.random.randint(96, size=(100))

        # dimensions = batch_size x 7 x 100
        t = targets[:, :, px, py]
        i = inputs[:, :, px, py]

        for i in range(100) :
            for j in range(100) :
                if(i != j and px[i]!=px[j] and py[i] != py[j]):
                    kl_loss += self.loss_pair(inputs, targets, py[i], px[i], py[j], px[j])
                    cnt += 1
        kl_loss /= cnt

        return (bg_loss + kl_loss)
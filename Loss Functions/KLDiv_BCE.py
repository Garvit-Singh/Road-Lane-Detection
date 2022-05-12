import torch
import torch.nn as nn

class KLDiv_BCE(nn.Module):
    def __init__(self):
        super(KLDiv_BCE, self).__init__()
    
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class OneOne(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OneOne, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.conv(x)

class NeuralNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, c=32, features=[16, 32, 64, 128],
    ):
        super(NeuralNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        self.finalconvs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of NeuralNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # bottleneck
        self.bottlenecks.append(DoubleConv(features[-1], features[-1]*2))       # 256 channels
        self.bottlenecks.append(OneOne(features[-1]*2, c))                      # c channels 

        # Up part of NeuralNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    c, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature, feature))
            self.ups.append(OneOne(feature, c))

        # final task specific
        self.finalconvs.append(OneOne(c, 32))
        self.finalconvs.append(OneOne(32, 7))
        self.finalconvs.append(nn.Softmax2d())

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            # concat_skip = torch.cat((skip_connection, x), dim=1)

            concat_skip = x + skip_connection
            x = self.ups[idx+1](concat_skip)
            x = self.ups[idx+2](x)
        
        for final_conv in self.finalconvs:
            x = final_conv(x)

        return x
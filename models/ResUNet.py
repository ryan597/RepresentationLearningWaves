###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie

###############################################################################

import torch
import torch.nn as nn


class ResNetBasicBlock(nn.Module):
    """
    Arguments:
        in_channels : (int) number of channels going into the block
        out_channels : (int) number of channels going out of the block
        *args : args to pass to the torch.nn.Conv2d function
        **kwargs : kwargs to pass to the torch.nn.Conv2d function

    Returns:
        A single ResNet Basic Block derived from the nn.Module class.

    The basic block is composed of 2 convolutions layered as follows:
        || SKIP_FROM              ||
        || conv 3x3 -> BN -> ReLU ||
        || conv 3x3 -> BN         ||
        || SKIP_TO                ||
        || ReLU                   ||
    The SKIP_FROM takes a copy of the input to the block and concatenates it
    to the input to the SKIP_TO layer. Batch Normalisation (BN) and the
    activation function ReLU are also applied in the layers.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs)
        )

    def forward(self, x):
        y = self.conv(x)
        x = torch.cat((y, x), dim=1)
        return nn.ReLU(x)


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, *args, **kwargs):
        self.blocks = nn.Sequential(
            ResNetBasicBlock(in_channels, out_channels)
        )

    def forward(self, x):
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)
        return x


class ResUNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

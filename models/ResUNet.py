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
        || SKIP_TO -> ConCat      ||
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


class ResNetEncodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, *args, **kwargs):
        self.blocks = nn.Sequential(
            ResNetBasicBlock(in_channels, out_channels),
            *[ResNetBasicBlock(out_channels, out_channels,
                               *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        return self.blocks(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=2, block_sizes=[64, 128, 256, 512, 1024],
                 depths=[2, 3, 5, 2]):
        self.in_out_sizes = list(zip(block_sizes[0:], block_sizes[1:]))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, block_sizes[0], kernel_size=7,
                      padding=3),
            nn.BatchNorm2d(block_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(block_sizes[0],  block_sizes[0],
                      kernel_size=3, padding=1)
        )
        self.encoder_layers = nn.ModuleList([
            ResNetEncodeLayer(in_channels, out_channels, n)
            for (in_channels, out_channels), n in zip(self.in_out_sizes,
                                                      depths)
        ])
        self.decoder_layers = nn.ModuleList([
            ResNetDecodeLayer()
        ])

    def forward(self, x):
        x = self.gate(x)
        skip_layers = []
        skip_layers.append(x)
        # encoder
        for layer, (in_channels, out_channels) in zip(self.encoder_layers,
                                                      self.in_out_sizes):
            # input to each layer is x, let y be the output of each layer
            # pooling before each layer as gate is applied first
            y = layer(nn.MaxPool2d(kernel_size=2, stride=2)(x))
            skip_layers.append(y)
            # apply 1x1 conv to have same number of channels at concatenation
            join_y = nn.Conv2d(in_channels, out_channels, kernel_size=1)(x)
            x = torch.cat((y, join_y), dim=1)
        # decoder
        for layer, (in_channels, out_channels) in zip(self.decoder_layers,
                                                      self.in_out_sizes[::-1]):
            # upsample
            # concat skip layers
            # decode layer
            y = layer(x)
            # concat 1x1 conv
        return x

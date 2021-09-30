"""
Class definitions for the ResUNet and its component layers.
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/DynamicTextureWaves

###############################################################################

# TO DO
# Fix encoder decoder
# issues with channels

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Arguments:
        in_channels (int): number of channels going into the block
        out_channels (int): number of channels going out of the block
        *args: args to pass to the torch.nn.Conv2d function
        **kwargs: kwargs to pass to the torch.nn.Conv2d function

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
                      *args, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        print(x.shape)
        return nn.ReLU()(x + self.conv(x))


class ResNetLayer(nn.Module):
    """
    Creates `n` of the ResNet basic blocks

    Arguments:
        in_channels (int): number of channels going into the layer
        out_channels: (int): number of channels going out of the layer
        n (int): number of blocks
        *args: args to pass to the torch.nn.Conv2d function
        **kwargs: kwargs to pass to the torch.nn.Conv2d function

    Metods:
        forward(self, x): x (torch.Tensor)

    Returns:
        A layer with `n` BasicBlocks derived from the nn.Module class
    """
    def __init__(self, in_channels, out_channels, n=1, *args, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            BasicBlock(in_channels, out_channels),
            *[BasicBlock(out_channels, out_channels,
                         *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        return self.blocks(x)


class ResUNet(nn.Module):
    """
    Class to build the ResUNet architecture.
    The blocks keep the channels constant while the 1x1 convolution will
    increase the channels after the layers.

    Arguments:
        in_channels (int): The number of input channels, default value 2
        block_sizes (list<int>): The channels in the following blocks.
            Default value [64, 128, 256, 512, 1024]
        depths (list<int>): The number of basic blocks to use at each layer.
            Default value [2, 3, 5, 2]

    Methods:
        forward(self, x): x (torch.Tensor) is the sample to compute the forward
            propagation throught the network. The tensor contains the batch
            samples first, then channels and height and width dimensions of the
            image.

    Returns:
        ResUNet object derived from nn.Module, can be trained in a standard
        PyTorch training loop.
    """
    def __init__(self, in_channels=2, block_sizes=[64, 128, 256, 512, 1024],
                 depths=[2, 3, 5, 2]):
        super().__init__()
        self.in_out_sizes = list(zip(block_sizes[0:], block_sizes[1:]))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, block_sizes[0], kernel_size=7,
                      padding=3),
            nn.BatchNorm2d(block_sizes[0]),
            nn.ReLU(),
            nn.Conv2d(block_sizes[0], block_sizes[0],
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(block_sizes[0]),
            nn.ReLU()
        )
        self.gate_conv = nn.Conv2d(block_sizes[0], block_sizes[1],
                                   kernel_size=1)
        self.encoder_layers = nn.ModuleList([
            ResNetLayer(in_channels, in_channels, n)
            for (in_channels, out_channels), n in zip(self.in_out_sizes[1:],
                                                      depths)
        ])
        self.decoder_layers = nn.ModuleList([
            ResNetLayer(in_channels, out_channels, n)
            for (in_channels, out_channels), n in zip(self.in_out_sizes[::-1],
                                                      [2 for _ in depths])
        ])  # decoder layers are fixed at depth 2

    def forward(self, x):
        """
        The forward propagation for the neural network defined by the __init__
        """
        x = self.gate(x)
        print(x.shape)
        x = self.gate_conv(x)
        print(x.shape)
        skip = []
        skip.append(x)
        # encoder
        for layer, (in_chan, out_chan) in zip(self.encoder_layers,
                                              self.in_out_sizes[1:]):
            # input to each layer is x, let y be the output of each layer
            # pooling before each layer as gate is applied first
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
            print(x.shape, "MaxPool")
            y = layer(x)
            print(y.shape, "Layer")
            skip.append(y)
            # apply 1x1 conv to end with out_chan
            join_xy = torch.cat((x, y), dim=1)
            x = nn.Conv2d(out_chan, out_chan, kernel_size=1)(join_xy)
        # decoder
        for i, layer, (in_chan, out_chan) in zip(range(len(skip)-1, -1, -1),
                                                 self.decoder_layers,
                                                 self.in_out_sizes[::-1]):
            # upsample
            x = nn.ConvTranspose2d(in_chan, out_chan,
                                   kernel_size=2, stride=2)(x)
            # concat with skip layers
            y = torch.cat((x, skip[i]), dim=1)
            # decode layer
            y = layer(x)
            # concat then 1x1 conv
            join_xy = torch.cat((x, y), dim=1)
            x = nn.Conv2d(in_chan, out_chan, kernel_size=1)(join_xy)
        return x

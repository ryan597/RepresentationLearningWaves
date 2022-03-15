"""
Class definitions for the ResUNet and its component layers.
"""

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
    def __init__(self, in_channels=2, out_channels=1,
                 block_sizes=[32, 64, 128, 256, 512, 1024],
                 depths=[2, 3, 5, 3, 2]):
        super().__init__()
        out_channels = out_channels
        in_out_sizes = list(zip(block_sizes, block_sizes[1:]))
        # Use a gate layer with kernel=7, wider receptive vision at start
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, block_sizes[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(block_sizes[0]),
            nn.ReLU(),
            nn.Conv2d(block_sizes[0], block_sizes[0], kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(block_sizes[0]),
            nn.ReLU()
        )
        encoder_layers = nn.ModuleList([
            ResNetLayer(out_channels, out_channels, n)
            for (in_channels, out_channels), n in zip(in_out_sizes,
                                                      depths)
        ])
        decoder_layers = nn.ModuleList([
            ResNetLayer(in_channels, in_channels, n)
            for (in_channels, out_channels), n in zip(in_out_sizes[::-1],
                                                      [2 for _ in depths])
        ])  # decoder layers are fixed at depth 2

        self.encode = nn.ModuleList([])  # All modules must be initialised here
        for layer, (in_chan, out_chan) in zip(encoder_layers, in_out_sizes):
            combinedlayer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                layer
                )
            self.encode.append(combinedlayer)

        self.decode_upsample = nn.ModuleList([])
        self.decode = nn.ModuleList([])
        for layer, (out_chan, in_chan) in zip(decoder_layers,
                                              in_out_sizes[::-1]):
            self.decode_upsample.append(nn.ConvTranspose2d(in_chan, out_chan,
                                        kernel_size=2, stride=2))
            combinedlayer = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                layer
            )
            self.decode.append(combinedlayer)

        self.decode.append(nn.Conv2d(in_out_sizes[0][0], 1, kernel_size=1))

    def forward(self, x):
        """
        The forward propagation for the neural network layers in the __init__()

        || gate(x)                ||
        || encoder(x)             ||
        || decoder(x)             ||
        || conv 1x1 (x)           ||
        || Sigmoid(x)             ||

        Args:
            x (torch.Tensor): The minibatch which is to be run through the
            network. Should be of shape (BATCH_SIZE, 2, HEIGHT, WIDTH), where
            BATCH_SIZE is the number of samples in each batch, HEIGHT and WIDTH
            are the pixel dimensions of the input samples.

        Returns:
            A torch.Tensor with shape of (BATCH_SIZE, 1, HEIGHT, WIDTH), where
            BATCH_SIZE is the number of samples in each batch, HEIGHT and WIDTH
            are the pixel dimensions of the input samples.
        """
        x = self.gate(x)
        skip = [x]
        # encoder
        for layer in self.encode:
            x = layer(x)
            skip.append(x)

        for t in skip:
            print(t.size())
        # decoder
        skip = skip[::-1][1:]  # Reverse skip for easy indexing, dont use first
        for i, upsample, layer in zip(range(len(skip)+1),
                                      self.decode_upsample,
                                      self.decode):
            x = upsample(x)

            x = torch.cat((x, skip[i]), dim=1)
            x = layer(x)

        x = self.decode[-1](x)
        return nn.Sigmoid()(x)

import copy

import torch
import torch.nn as nn
import torchvision.models as TVmodels


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x) + x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, *args, **kwargs):
        super().__init__()
        self.downsample = nn.Sequential(
            BasicBlock(in_channels, out_channels),
            *[BasicBlock(out_channels, out_channels, *args, **kwargs)
                for _ in range(n-1)]
        )

    def forward(self, x):
        return self.downsample(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample(x)


class Decoder(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__()
        self.decode = nn.Sequential(
            *[UpsampleBlock(in_ch, out_ch) for [in_ch, out_ch] in channels]
        )

    def forward(self, x):
        return self.decode(x)


class ResNet_backbone(nn.Module):
    def __init__(self, layers=50, freeze=5, masks=False, dual=False):
        super().__init__()
        self.masks = masks
        self.dual = dual
        if layers == 50:
            channels = [[2048 + 2048*dual, 512], [512, 64], [64, 1]]
            backbone = TVmodels.resnet.resnet50(
                pretrained=False,
                replace_stride_with_dilation=[False, True, True])
            backbone.load_state_dict(
                torch.load("weights/resnet50-0676ba61.pth"))
        elif layers == 18:
            channels = [[512 + 512*dual, 256],
                        [256, 128], [128, 64], [64, 1]]
            backbone = TVmodels.resnet.resnet18(
                pretrained=False)
            backbone.load_state_dict(
                torch.load("weights/resnet18-f37072fd.pth"))
        # Remove the final two layers (avgpool and fc (fully connected))
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        count = 0  # Freeze early layers
        for child in backbone.children():
            count += 1
            if count < freeze:
                for param in child.parameters():
                    param.requires_grad = False

        if not dual:
            self.backbone = backbone
            self.decode = Decoder(channels)
        else:
            self.backbone1 = copy.deepcopy(backbone)
            self.backbone2 = copy.deepcopy(backbone)
            self.decode = Decoder(channels)

        # Freeze the decoder if masks
        # if self.masks:
        #    for child in list(self.decode.decode)[:-1]:
        #       for param in child.parameters():
        #            param.requires_grad = False

    def forward(self, x):
        if not self.dual:
            x = self.backbone(x)
            x = self.decode(x)
        else:
            img1 = x[:, 0:3]
            img2 = x[:, 3:6]
            x1 = self.backbone1(img1)
            x2 = self.backbone2(img2)
            x = self.decode(torch.cat((x1, x2), dim=1))
        if self.masks:
            x = torch.cat((x.softmax(dim=0), 1 - x.softmax(dim=0)), dim=1)
        return x


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
    def __init__(self, in_channels=2, out_channels=1, masks=True, freeze=False,
                 block_sizes=[32, 64, 128, 256, 512, 1024],
                 depths=[2, 3, 5, 3, 2]):
        super().__init__()
        self.masks = masks
        self.freeze = freeze
        out_channels = out_channels
        in_out_sizes = list(zip(block_sizes, block_sizes[1:]))
        # Use a gate layer with kernel=7, wider receptive vision at start
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, block_sizes[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(block_sizes[0]),
            nn.Mish(),
            nn.Conv2d(block_sizes[0], block_sizes[0], kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(block_sizes[0]),
            nn.Mish()
        )
        encoder_layers = nn.ModuleList([
            ResidualLayer(out_channels, out_channels, n)
            for (in_channels, out_channels), n in zip(in_out_sizes,
                                                      depths)
        ])
        decoder_layers = nn.ModuleList([
            ResidualLayer(in_channels, in_channels, n)
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

        if self.masks:
            for child in self.encode.children():
                for param in child:
                    param.requires_grad = False

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

        # decoder
        skip = skip[::-1][1:]  # Reverse skip for easy indexing, dont use first
        for i, upsample, layer in zip(range(len(skip)+1),
                                      self.decode_upsample,
                                      self.decode):
            x = upsample(x)

            x = torch.cat((x, skip[i]), dim=1)
            x = layer(x)

        x = self.decode[-1](x)
        if self.masks:
            x = torch.cat((x.softmax(dim=0), 1 - x.softmax(dim=0)), dim=1)
        return x

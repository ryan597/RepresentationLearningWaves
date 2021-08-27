###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie

###############################################################################

import torch.nn as nn


class DoubleConv(nn.Module):
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
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_size, *args, **kwargs):
        super().__init__()
        down_sizes = [in_channels, *layer_size]
        up_sizes = [*layer_size[::-1], out_channels]

        self.conv_down = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in
            zip(down_sizes, down_sizes[1:])])

        self.down_sample = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for i in
            range(len(layer_size)-1)])

        self.up_sample = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2) for
            in_ch, out_ch in zip(up_sizes, up_sizes[1:-1])])

        self.conv_up = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in
            zip(up_sizes, up_sizes[1:])])

    def forward(self, x):
        copies = []
        for i, conv in enumerate(self.conv_down[:-1]):
            x = conv(x)
            copies.append(x)
            x = self.down_sample[i](x)

        x = self.conv_down[-1](x)
        for i, conv in enumerate(self.conv_up[:-1]):
            x = self.up_sample[i](x)
            x = conv(torch.cat((x, copies[-(i+1)]), dim=1))

        x = self.conv_up[-1](x)
        return x

import copy

import torch
import torch.nn as nn
import torchvision.models as TVmodels


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gate=False):
        super().__init__()
        self.gate = gate
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        if self.gate:
            x = self.conv(x)
        else:
            x = self.conv(x) + x
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super().__init__()
        self.layer = nn.Sequential(
            ResBlock(in_channels, out_channels, gate=True),
            *[ResBlock(out_channels, out_channels)
                for _ in range(n - 1)]
        )

    def forward(self, x):
        return self.layer(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels)
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
    def __init__(self, layers=50, freeze=5, masks=False):
        super().__init__()
        self.masks = masks
        if layers == 50:
            channels = [[4096, 512], [512, 64], [64, 1 + self.masks]]
            backbone = TVmodels.resnet.resnet50(
                pretrained=False,
                replace_stride_with_dilation=[False, True, True])
            backbone.load_state_dict(
                torch.load("weights/resnet50-0676ba61.pth"))
        elif layers == 18:
            channels = [[1024, 256], [256, 128], [128, 64], [64, 1 + self.masks]]
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

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)
        self.decode = Decoder(channels)

    def forward(self, x):
        img1 = x[:, 0:3]
        img2 = x[:, 3:6]
        x1 = self.backbone1(img1)
        x2 = self.backbone2(img2)
        x = self.decode(torch.cat((x1, x2), dim=1))
        if self.masks:
            x = torch.softmax(x, dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, pretrain_bn=False):
        super().__init__()
        self.pretrain_bn = pretrain_bn

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.Conv1 = ResidualLayer(img_ch, 64, 2)
        self.Conv2 = ResidualLayer(64, 128, 2)
        self.Conv3 = ResidualLayer(128, 256, 2)
        self.Conv4 = ResidualLayer(256, 512, 2)
        self.Conv5 = ResidualLayer(512, 1024, 2)

        self.Up5 = UpConv(1024, 512)
        self.UpConv5 = ResidualLayer(1024, 512, 1)

        self.Up4 = UpConv(512, 256)
        self.UpConv4 = ResidualLayer(512, 256, 1)

        self.Up3 = UpConv(256, 128)
        self.UpConv3 = ResidualLayer(256, 128, 1)

        self.Up2 = UpConv(128, 64)
        self.UpConv2 = ResidualLayer(128, 64, 1)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        if self.pretrain_bn:
            skip4 = e4 * 0
            skip3 = e3 * 0
            skip2 = e2 * 0
            skip1 = e1 * 0

        d5 = self.Up5(e5)
        d5 = torch.cat((skip4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((skip3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((skip2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((skip1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, pretrain_bn=False):
        super().__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        if pretrain_bn:
            self.skip_connection_gradients(require_grad=False)

    def skip_connection_gradients(self, require_grad):
        for param in self.Att5.W_x.parameters():
            param.requires_grad = require_grad
        for param in self.Att4.W_x.parameters():
            param.requires_grad = require_grad
        for param in self.Att3.W_x.parameters():
            param.requires_grad = require_grad
        for param in self.Att2.W_x.parameters():
            param.requires_grad = require_grad

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        return out

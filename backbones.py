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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x) + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample(x)


class Decoder(nn.Module):
    def __init__(self, channels=[[2048, 512], [512, 64], [64, 2]],
                 dual=False, *args, **kwargs):
        super().__init__()
        if dual:
            channels[0][0] = 4096

        self.decode = nn.Sequential(
            *[UpsampleBlock(in_ch, out_ch) for [in_ch, out_ch] in channels]
        )

    def forward(self, x):
        return self.decode(x)


class ResNet_backbone(nn.Module):
    def __init__(self, layers=50, freeze=5, dual=False):
        super().__init__()
        self.dual = dual
        if layers == 50:
            backbone = TVmodels.resnet.resnet50(
                pretrained=False,
                replace_stride_with_dilation=[False, True, True])
            backbone.load_state_dict(
                torch.load("weights/resnet50-0676ba61.pth"))
        elif layers == 18:
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
            self.decode = Decoder()
        else:
            self.backbone1 = copy.deepcopy(backbone)
            self.backbone2 = copy.deepcopy(backbone)
            self.decode = Decoder(dual=True)

    def forward(self, x):
        if not self.dual:
            x = self.backbone(x)
            return self.decode(x)
        else:
            img1 = x[:, 0:3]
            img2 = x[:, 3:6]
            x1 = self.backbone1(img1)
            x2 = self.backbone2(img2)
            return self.decode(torch.cat((x1, x2), dim=1))

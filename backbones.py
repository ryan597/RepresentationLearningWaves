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
            # nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      *args, **kwargs),
            # nn.BatchNorm2d(out_channels),
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
        if self.masks:
            for child in list(self.decode.decode)[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

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

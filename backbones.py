import torch
import torch.nn as nn
import torchvision.models as TVmodels
import torchvision.models.segmentation.fcn as FCN


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
                pretrained=False,
                replace_stride_with_dilation=[False, True, True])
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
            self.fcnhead = FCN.FCNHead(2048, 1)

        else:
            self.backbone1 = backbone
            self.backbone2 = backbone
            self.fcnhead = FCN.FCNHead(4096, 1)

    def forward(self, x):
        if not self.dual:
            x = self.backbone(x)
            return self.fcnhead(x)
        else:
            img1 = x[:, 0]
            img2 = x[:, 1]
            x1 = self.backbone1(img1)
            x2 = self.backbone2(img2)
            return self.fcnhead(torch.cat((x1, x2), dim=1))

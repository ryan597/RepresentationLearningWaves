import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torchvision.models.segmentation.fcn as FCN
from torchvision.ops import sigmoid_focal_loss
import matplotlib.pyplot as plt
import data_utils

# Randomness must be disabled for distributed training!
pl.utilities.seed.seed_everything(42)

class Res50(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        backbone = torchvision.models.resnet.resnet50(
            pretrained=False, replace_stride_with_dilation=[False, True, True])
        # resnet50-0676ba61.pth # resnet18-f37072fd.pth
        backbone.load_state_dict(
            torch.load("models/weights/resnet50-0676ba61.pth"))

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        ct = 0
        for child in self.backbone.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        self.fcnhead = FCN.FCNHead(2048, num_outputs)

    def forward(self, x):
        img1 = x[:, 0]
        img2 = x[:, 1]
        x1 = self.backbone(img1)
        x2 = self.backbone(img2)
        return self.fcnhead(torch.cat((x1, x2), dim=1))




class ResUNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        backbone = torchvision.models.resnet.resnet50(
            pretrained=False, replace_stride_with_dilation=[False, True, True])
        # resnet50-0676ba61.pth # resnet18-f37072fd.pth
        backbone.load_state_dict(
            torch.load("models/weights/resnet50-0676ba61.pth"))

        backbone = nn.Sequential(*list(backbone.children())[:-2])
        ct = 0
        for child in backbone.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        self.fcnhead = FCN.FCNHead(2048, num_outputs)

        self.backbone1 = backbone  # two networks that aren't sharing weights
        self.backbone2 = backbone
        self.fcnhead = FCN.FCNHead(4096, num_outputs)

    def forward(self, x):
        img1 = x[:, 0]
        img2 = x[:, 1]
        x1 = self.backbone1(img1)
        x2 = self.backbone2(img2)
        return self.fcnhead(torch.cat((x1, x2), dim=1))


class LitModel(pl.LightningModule):
    def __init__(self, base_model, lr, train_path, valid_path,
                 image_shape=(512, 1024), batch_size=10, shuffle=True):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.train_path = train_path
        self.valid_path = valid_path
        self.masks=True
        self.input_N=1
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.criterion = sigmoid_focal_loss
        self.save_hyperparameters(ignore=['base_model'])

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            verbose=True)
        lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
                }
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config
                }

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        if batch_idx == 2:
            self.save_outputs(outputs, inputs, labels, 'training', batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels, reduction='mean')
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

    def save_outputs(self, outputs, inputs, labels, location, batch_idx):
        fig, ax = plt.subplots(5,  # give 5 outputs | rows
                               4,  # input, ground truth, pred, diff | cols
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})
        for i, [input_image, pred_image, gt_image] in enumerate(zip(
                inputs, outputs, labels)):
            ax[i, 0].imshow(input_image.detach().cpu().numpy()[0], cmap='gray')
            ax[i, 1].imshow(gt_image.detach().cpu().numpy()[1], cmap='gray')
            ax[i, 2].imshow(pred_image.detach().cpu().numpy()[1], cmap='gray')
            ax[i, 3].imshow(np.abs((gt_image.detach().cpu().numpy()[1] - pred_image.detach().cpu().numpy()[1])), cmap='gray')
            if i == 4:
                break
        fig.suptitle("Model Predictions - Segmentation", fontsize=15)
        ax[0, 0].set_title("Input", fontsize=10)
        ax[0, 1].set_title("Ground Truth", fontsize=10)
        ax[0, 2].set_title("Prediction", fontsize=10)
        ax[0, 3].set_title("Difference", fontsize=10)

        plt.savefig(f"outputs/figures/{location}/{self.current_epoch}-{batch_idx}.png")
        plt.close()

    def test_step(self, batch, batch_idx):
        test_loss = 0
        self.log("test_loss", test_loss, on_epoch=True, sync_dist=True)

    def train_dataloader(self):
        return data_utils.load_data(
            path=self.train_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            masks=self.masks,
            input_N=self.input_N
        )

    def val_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            input_N=self.input_N
        )

    def test_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            input_N=self.input_N
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",
                        help="Path to directory of training datasets")
    parser.add_argument("--valid_path",
                        help="Path to directory of validation datasets",
                        default=None)
    parser.add_argument("--test_path",
                        help="Path to directory of testing datasets",
                        default=None)
    parser.add_argument("--backbone",
                        help="Backbone of model, resnet50 or resnet18")
    parser.add_argument("--masks",
                        help="Train for segmentation or frame prediciton")
    parser.add_argument("--checkpoint",
                        help="Path to checkpoint", default=False)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)

    if args.masks == 'True':
        outputs = 2
    else:
        outputs = 1

    # compute nodes have no internet access so manually load weights
    if args.backbone == 'resnet50':
        backbone = torchvision.models.resnet.resnet50(pretrained=False)
        backbone.load_state_dict(
            torch.load("models/weights/resnet50-0676ba61.pth"))

        ct = 0
        for child in backbone.children():
            # ct += 1
            # if ct < 8:
            for param in child.parameters():
                param.requires_grad = False
        model = FCN._fcn_resnet(backbone=backbone, num_classes=2, aux=True)

    elif args.backbone == 'resnet18':

        model = ResUNet(outputs)

    if args.checkpoint:
        model = model.load_from_checkpoint(args.checkpoint)

    else:
        model = LitModel(model,
                         0.0001,
                         args.train_path,
                         args.valid_path
                         )

    trainer.fit(model)

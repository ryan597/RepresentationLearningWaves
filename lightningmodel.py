import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchvision.ops import sigmoid_focal_loss

import data_utils


class LightningModel(pl.LightningModule):
    def __init__(self, base_model, lr, train_path, valid_path,
                 image_shape=(512, 1024), batch_size=10, shuffle=True,
                 masks=False, dual=False):
        super().__init__()
        self.model = base_model
        self.train_path = train_path
        self.valid_path = valid_path
        self.image_shape = image_shape
        self.lr = lr
        self.masks = masks
        self.dual = dual
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.criterion = sigmoid_focal_loss if masks else nn.L1Loss()
        self.save_hyperparameters(ignore=['base_model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            verbose=True)
        lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "monitor": "train_loss_on_epoch",
                }
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config
                }

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            loss = self.criterion(outputs, labels, reduction='mean')
        else:
            loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        if batch_idx == 2:
            self.save_outputs(outputs, inputs, labels, 'training', batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            val_loss = self.criterion(outputs, labels, reduction='mean')
        else:
            val_loss = self.criterion(outputs, labels)
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

    def save_outputs(self, outputs, inputs, labels, loc, batch_idx):
        fig, ax = plt.subplots(5,  # give 5 outputs | rows
                               4,  # input, ground truth, pred, diff | cols
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})
        for i, [input_image, pred_image, gt_image] in enumerate(zip(
                inputs, outputs, labels)):
            input_image = input_image.detach().cpu().numpy()[0]
            gt_image = gt_image.detach().cpu().numpy()[1]
            pred_image = pred_image.detach().cpu().numpy()[1]
            ax[i, 0].imshow(input_image, cmap='gray')
            ax[i, 1].imshow(gt_image, cmap='gray')
            ax[i, 2].imshow(pred_image, cmap='gray')
            ax[i, 3].imshow(np.abs((gt_image - pred_image)), cmap='gray')
            if i == 4:
                break
        title = "Segmentation" if self.masks else "Frame Prediciton"
        fig.suptitle(f"Model Outputs - {title}", fontsize=13)
        ax[0, 0].set_title("Input", fontsize=10)
        ax[0, 1].set_title("Ground Truth", fontsize=10)
        ax[0, 2].set_title("Prediction", fontsize=10)
        ax[0, 3].set_title("Difference", fontsize=10)

        save_path = f"outputs/figures/{loc}/{self.current_epoch}-{batch_idx}"
        plt.savefig(save_path + ".png")
        plt.close()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            test_loss = self.criterion(outputs, labels, reduction='mean')
        else:
            test_loss = self.criterion(outputs, labels)
        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.save_outputs(outputs, inputs, labels, 'test', batch_idx)

    def train_dataloader(self):
        return data_utils.load_data(
            path=self.train_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            masks=self.masks,
            dual=self.dual
        )

    def val_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            dual=self.dual
        )

    def test_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            dual=self.dual
        )

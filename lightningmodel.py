import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
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
        self.criterion = sigmoid_focal_loss if masks else F.l1_loss
        self.save_hyperparameters(ignore=['base_model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            verbose=True)
        lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
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

        # if self.masks:
        #     output = outputs.detach().cpu().numpy()
        #     label = labels.detach().cpu().numpy()
        #     pixelacc = np.sum((output[0] > 0.3) == label[0]) * 100 / np.size(output[0])
        #
        #     # IoU of averaged over FG and BG
        #     output[0] = output[0] > 0.3  # Threshold the FG prob at 0.3
        #     output[1] = output[0] > 0.7
        #     inter = np.sum(output * label, axis=1)
        #     union = np.sum(output, axis=1) + np.sum(label, axis=1) - inter
        #     iou = np.mean((inter + 1) / (union + 1))
        #
        #     # Dice coef
        #     # factor of 2 cancels from 2 channels and average
        #     dc = np.sum(2 * inter / np.size(output))
        #
        #     self.log('pixelacc', pixelacc, on_step=False, on_epoch=True,
        #              prog_bar=True, logger=True)
        #     self.log('IoU', iou, on_step=False, on_epoch=True,
        #              prog_bar=True, logger=True)
        #     self.log('Dice Coef', dc, on_step=False, on_epoch=True,
        #              prog_bar=True, logger=True)

        if batch_idx == 2:
            self.save_outputs(outputs, inputs, labels, 'training', batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels, reduction='mean')
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)

        if self.masks:
            output = outputs.detach().cpu().numpy()
            label = labels.detach().cpu().numpy()
            pixelacc = np.sum((output[0] > 0.3) == label[0]) * 100 / np.size(output[0])

            # IoU of averaged over FG and BG
            output[0] = output[0] > 0.3  # Threshold the FG prob at 0.3
            output[1] = output[0] > 0.7
            inter = np.sum(output * label, axis=1)
            union = np.sum(output, axis=1) + np.sum(label, axis=1) - inter
            iou = np.mean((inter + 1) / (union + 1))

            # Dice coef
            # factor of 2 cancels from 2 channels and average
            dc = np.sum(2 * inter / np.size(output))

            self.log('pixelacc', pixelacc, on_epoch=True, sync_dist=True,
                     prog_bar=True, logger=True)
            self.log('IoU', iou, on_epoch=True, sync_dist=True,
                     prog_bar=True, logger=True)
            self.log('Dice Coef', dc, on_epoch=True, sync_dist=True,
                     prog_bar=True, logger=True)

        if batch_idx == 2:
            self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

    def save_outputs(self, outputs, inputs, labels, loc, batch_idx):
        fig, ax = plt.subplots(5,  # give 5 outputs | rows
                               4,  # input, ground truth, pred, diff | cols
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})
        for i, [input_image, pred_image, gt_image] in enumerate(zip(
                inputs, outputs, labels)):
            input_image = input_image.detach().cpu().numpy()[1]
            gt_image = gt_image.detach().cpu().numpy()[0]
            pred_image = pred_image.detach().cpu().numpy()[0]
            cmap = 'gray'
            ax[i, 0].imshow(input_image, cmap=cmap)
            ax[i, 1].imshow(gt_image, cmap=cmap)

            if self.masks:  # Threshold if pred should be a mask
                title = "Segmentation"
                prob_diff = "Probability Map"
                subdir = "masks"
                thresh_image = pred_image >= 0.3
                ax[i, 2].imshow(thresh_image, cmap=cmap)
                ax[i, 3].imshow(pred_image, cmap=cmap)
            else:
                title = "Frame Prediciton"
                prob_diff = "Difference"
                subdir = "frames"
                ax[i, 2].imshow(pred_image, cmap=cmap)
                ax[i, 3].imshow(np.abs((gt_image - pred_image)), cmap=cmap)
            if i == 4:
                break
        fig.suptitle(f"Model Outputs - {title}", fontsize=13)
        ax[0, 0].set_title("Input", fontsize=10)
        ax[0, 1].set_title("Ground Truth", fontsize=10)
        ax[0, 2].set_title("Prediction", fontsize=10)
        ax[0, 3].set_title(f"{prob_diff}", fontsize=10)

        save_path = f"outputs/figures/{loc}/{subdir}/{self.current_epoch}-{batch_idx}"
        plt.savefig(save_path + ".png", dpi=600)
        plt.close()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = self.criterion(outputs, labels, reduction='mean')
        output = outputs.detach().cpu().numpy()
        label = labels.detach().cpu().numpy()
        pixelacc = np.sum((output[0] > 0.3) == label[0]) * 100 / np.size(output[0])

        # IoU of averaged over FG and BG
        output[0] = output[0] > 0.3  # Threshold the FG prob at 0.3
        output[1] = output[0] > 0.7
        inter = np.sum(output * label, axis=1)
        union = np.sum(output, axis=1) + np.sum(label, axis=1) - inter
        iou = np.mean((inter + 1) / (union + 1))

        # Dice coef
        # factor of 2 cancels from 2 channels and average
        dc = np.sum(2 * inter / np.size(output))

        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.log('pixelacc', pixelacc, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.log('IoU', iou, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.log('Dice Coef', dc, on_epoch=True, sync_dist=True,
                 prog_bar=True, logger=True)
        self.save_outputs(outputs, inputs, labels, 'test', batch_idx)

    def train_dataloader(self):
        return data_utils.load_data(
            path=self.train_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            masks=self.masks,
            dual=self.dual,
            aug=True
            )

    def val_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            dual=self.dual,
            aug=False
        )

    def test_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            dual=self.dual,
            aug=False
        )

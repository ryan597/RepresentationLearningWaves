import os

import matplotlib.pyplot as plt
import numpy as np
# import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.functional import dice, jaccard_index

import data_utils


def maskedL1loss(output, target, inputs, reduction='mean'):
    mask = torch.abs(inputs[0] - inputs[-1])
    loss = torch.abs(output - target)
    loss = (mask**2) * loss
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


class LightningModel(pl.LightningModule):
    def __init__(self, base_model, lr, train_path, valid_path,
                 image_shape=(512, 1024), batch_size=10, shuffle=True,
                 masks=False, seq_length=2, channels=1, step=1, thresh=0.2):
        super().__init__()
        self.model = base_model
        self.train_path = train_path
        self.valid_path = valid_path
        self.image_shape = image_shape
        self.lr = lr
        self.masks = masks
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.channels = channels
        self.step = step
        self.criterion = sigmoid_focal_loss if masks else maskedL1loss
        self.thresh = thresh
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
        if self.masks:
            loss = self.criterion(outputs, labels, reduction='mean')
        else:
            loss = self.criterion(outputs, labels, inputs)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.masks:
            output = outputs.detach().cpu()
            label = labels.detach().cpu()

            # Threshold the FG prob
            output[:, 0] = output[:, 0] > self.thresh
            output[:, 1] = 1 - output[:, 0]
            iou = jaccard_index(output.int(), label.int(), average="macro", num_classes=2)
            dc = dice(output.int(), label.int(), average="macro", num_classes=2)

            self.log('IoU', iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('Dice', dc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx in [1, 2, 3, 4, 5]:  # check some random batches
            self.save_outputs(outputs, inputs, labels, 'training', batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            val_loss = self.criterion(outputs, labels, reduction='mean')
        else:
            val_loss = self.criterion(outputs, labels, inputs)
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        if self.masks:
            output = outputs.detach().cpu()
            label = labels.detach().cpu()

            # Threshold the FG prob at 0.2
            output[:, 0] = output[:, 0] > self.thresh
            output[:, 1] = 1 - output[:, 0]
            iou = jaccard_index(output.int(), label.int(), average="macro", num_classes=2)
            dc = dice(output.int(), label.int(), average="macro", num_classes=2)

            self.log('val_IoU', iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_Dice', dc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx in [50, 150, 250, 300, 500] or self.masks:  # check some batches
            self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

    def save_outputs(self, outputs, inputs, labels, loc, batch_idx):
        fig, ax = plt.subplots(5,  # give 5 outputs | rows
                               4,  # input, ground truth, pred, diff | cols
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})
        for i, [input_image, pred_image, gt_image] in enumerate(zip(
                inputs, outputs, labels)):
            input_image = input_image.detach().cpu().numpy()[-1]
            gt_image = gt_image.detach().cpu().numpy()[0]
            pred_image = pred_image.detach().cpu().numpy()[0]
            cmap = 'gray'
            ax[i, 0].imshow(input_image, cmap=cmap)
            ax[i, 1].imshow(gt_image, cmap=cmap)

            if self.masks:  # Threshold if pred should be a mask
                title = "Segmentation"
                prob_diff = "Probability Map"
                subdir = "masks"
                thresh_image = pred_image >= 0.2
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

        save_path = f"outputs/figures/{loc}/{subdir}/" + \
                    f"{self.current_epoch}-{batch_idx}_{os.environ['SLURM_JOB_ID']}"
        plt.savefig(save_path + ".png", dpi=600)
        plt.close()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = self.criterion(outputs, labels, reduction='mean')
        output = outputs.detach().cpu()
        label = labels.detach().cpu()

        # Threshold the FG prob
        output[:, 0] = output[:, 0] > self.thresh
        output[:, 1] = 1 - output[:, 0]
        iou = jaccard_index(output.int(), label.int(), average="macro", num_classes=2)
        dc = dice(output.int(), label.int(), average="macro", num_classes=2)

        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('test_IoU', iou, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('test_Dice', dc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        self.save_outputs(outputs, inputs, labels, 'test', batch_idx)

    def train_dataloader(self):
        return data_utils.load_data(
            path=self.train_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            masks=self.masks,
            seq_length=self.seq_length,
            step=self.step,
            aug=True,
            channels=self.channels
        )

    def val_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            seq_length=self.seq_length,
            step=self.step,
            aug=False,
            channels=self.channels
        )

    def test_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=False,
            masks=self.masks,
            seq_length=self.seq_length,
            step=self.step,
            aug=False,
            channels=self.channels
        )

    def on_load_checkpoint(self, checkpoint):
        # Hack for size mis-match in state_dict
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    # logger.info(f"Skip loading parameter: {k}, "
                    #             f"required shape: {model_state_dict[k].shape}, "
                    #             f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                # self.logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

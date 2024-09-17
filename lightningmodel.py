import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.functional import dice, jaccard_index
from sklearn.metrics import precision_score, recall_score, brier_score_loss

import data_utils


def maskedL1loss(output, target, inputs, reduction='mean'):
    mask = torch.abs(target - inputs[-1]) > 0.085
    loss = F.l1_loss(output, target, reduction='none')
    loss = (mask * 10 + 1) * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class LightningModel(pl.LightningModule):
    def __init__(self, base_model, lr, train_path, valid_path,
                 image_shape=(512, 1024), batch_size=10, shuffle=True,
                 masks=False, seq_length=2, channels=1, step=1, thresh=0.5):
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
        self.criterion = sigmoid_focal_loss if masks else F.mse_loss  # maskedL1loss
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
            threshold_mode="abs",
            patience=5,
            verbose=True)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "monitor": "train_loss_epoch",
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
            loss = self.criterion(outputs, labels, reduction='mean', alpha=0.032, gamma=2)
        else:
            loss = self.criterion(outputs, labels, reduction="mean")

        self.log("train_loss", loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Debugging
        # if self.masks and self.current_epoch % 5 == 0:
        #    output = outputs.detach().cpu()
        #    label = labels.detach().cpu()
        #
        #    # Threshold the FG prob
        #    brier = brier_score_loss(label[:, 0].flatten().int(), output[:, 0].flatten())
        #    output = output[:, 0] > self.thresh
        #    label = label[:, 0]
        #    iou = jaccard_index(output.int(), label.int(), average=None, num_classes=2)[0]
        #    dc = dice(output.int(), label.int(), average=None, num_classes=2)[0]
        #    precision = precision_score(label.flatten().int(), output.flatten().int(), zero_division=0)
        #    recall = recall_score(label.flatten().int(), output.flatten().int(), zero_division=0)
        #
        #    self.log('IoU', iou, prog_bar=True, logger=True, sync_dist=False)
        #    self.log('Dice', dc, prog_bar=True, logger=True, sync_dist=False)
        #    self.log('P', precision, prog_bar=True, logger=True, sync_dist=False)
        #    self.log('R', recall, prog_bar=True, logger=True, sync_dist=False)
        #    self.log('B', brier, prog_bar=True, logger=True, sync_dist=False)

        if batch_idx in [1, 2, 3, 4, 5]:  # check some random batches
            self.save_outputs(outputs, inputs, labels, 'training', batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            val_loss = self.criterion(outputs, labels, reduction='mean', alpha=0.032, gamma=2)
        else:
            val_loss = self.criterion(outputs, labels, reduction="mean")
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        # Monitor metrics in training, disable for speedup
        #if self.masks:
        #    output = outputs.detach().cpu()
        #    label = labels.detach().cpu()

        #    brier = brier_score_loss(label[:, 0].flatten().int(), output[:, 0].flatten())
        #    # Threshold the FG prob
        #    output[:, 0] = output[:, 0] > self.thresh
        #    output[:, 1] = 1 - output[:, 0]
        #    iou = jaccard_index(output.int(), label.int(), task='binary', average=None, num_classes=2)
        #    dc = dice(output.int(), label.int(), average=None, num_classes=2)
        #    precision = precision_score(label[:, 0].flatten().int(), output[:, 0].flatten().int(), zero_division=0)
        #    recall = recall_score(label[:, 0].flatten().int(), output[:, 0].flatten().int(), zero_division=0)

        #    self.log('val_IoU', iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #    self.log('val_Dice', dc[0], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #    self.log('val_Dice_bg', dc[1], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #    self.log('val_P', precision, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #    self.log('val_R', recall, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #    self.log('val_B', brier, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        #if torch.rand(1) > 0.7:  # check some batches
        #    self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

    def on_validation_end(self):
        if not self.masks and (self.current_epoch + 1) >= 20:
            # Allow skip connections
            self.model.pretrain_bn = False

    def save_outputs(self, outputs, inputs, labels, loc, batch_idx):
        fig, ax = plt.subplots(5,  # give 5 outputs | rows
                               4,  # input, ground truth, pred, diff | cols
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []},
                               figsize=(15, 10))
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
                thresh_pred = np.ma.masked_where(pred_image < self.thresh, pred_image)
                ax[i, 2].imshow(input_image, cmap=cmap)
                ax[i, 2].imshow(thresh_pred, cmap='Reds', interpolation=None, alpha=0.7)
                ax[i, 3].imshow(pred_image, cmap=cmap)
            else:
                title = "Frame Prediciton"
                prob_diff = "Difference"
                subdir = "frames"
                ax[i, 2].imshow(pred_image, cmap=cmap)
                ax[i, 3].imshow(np.abs((gt_image - pred_image)), cmap=cmap)
            if i == 4:
                break
        fig.suptitle(f"Model Outputs - {title}", fontsize=20)
        ax[0, 0].set_title("Input", fontsize=15)
        ax[0, 1].set_title("Ground Truth", fontsize=15)
        ax[0, 2].set_title("Prediction", fontsize=15)
        ax[0, 3].set_title(f"{prob_diff}", fontsize=15)

        save_path = f"../scratch/outputs/figures/{loc}/{subdir}/{os.environ['SLURM_JOB_ID']}/" + \
                    f"{self.current_epoch}-{batch_idx}"
        plt.savefig(save_path + ".png", dpi=300)  # increase dpi for better quality figures (reduced to save space)
        plt.close()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        if self.masks:
            test_loss = self.criterion(outputs, labels, reduction='mean', alpha=0.032, gamma=2)
        else:
            test_loss = self.criterion(outputs, labels, reduction="mean")

        output = outputs.detach().cpu()
        label = labels.detach().cpu()

        if self.masks:
            brier = brier_score_loss(label[:, 0].flatten().int(), output[:, 0].flatten())
            # Threshold the FG prob
            output[:, 0] = output[:, 0] > self.thresh
            output[:, 1] = 1 - output[:, 0]
            iou = jaccard_index(output.int(), label.int(), task='binary', average=None, num_classes=2)
            dc = dice(output.int(), label.int(), average=None, num_classes=2)
            precision = precision_score(label[:, 0].flatten().int(), output[:, 0].flatten().int(), zero_division=0)
            recall = recall_score(label[:, 0].flatten().int(), output[:, 0].flatten().int(), zero_division=0)

            self.log('IoU', iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('Dice', dc[0], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('Dice_bg', dc[1], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('P', precision, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('R', recall, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('Brier', brier, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.save_outputs(outputs, inputs, labels, 'validation', batch_idx)

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': test_loss}


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


def reset_all_weights(model):
    @torch.no_grad()
    def weight_reset(m):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    # Apply recursively
    model.apply(fn=weight_reset)

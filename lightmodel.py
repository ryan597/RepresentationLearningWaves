import argparse
import pytorch_lightning as pl
import torch
import torchvision
import data_utils

# Randomness must be disabled for distributed training!
pl.utilities.seed.seed_everything(42)


class LitModel(pl.LightningModule):
    def __init__(self, base_model, lr, train_path, valid_path,
                 image_shape=(256, 512), batch_size=10, shuffle=True):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.train_path = train_path
        self.valid_path = valid_path
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
            threshold=0.001,
            verbose=True)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # augmentation here?
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = 0
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        test_loss = 0
        self.log("test_loss", test_loss)

    def train_dataloader(self):
        return data_utils.load_data(
            path=self.train_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

    def test_dataloader(self):
        return data_utils.load_data(
            path=self.valid_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_path",
        help="Path to directory of training datasets")
    parser.add_argument("-v", "--valid_path",
        help="Path to directory of validation datasets", default=None)
    parser.add_argument("--test_path",
        help="Path to directory of testing datasets", default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)

    base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False, num_classes=2)
    base_model.load_state_dict(torch.load("models/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))

    model = LitModel(base_model,
                     0.001,
                     args.train_path,
                     args.valid_path
                     )

    trainer.fit(model)

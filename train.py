from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torchvision.models.segmentation import fcn_resnet50

from AttnUnet import AttentionUNet
from backbones import ResNet_backbone, ResUNet
from lightningmodel import LightningModel


def main(hparams, *args):
    pl.utilities.seed.seed_everything(2022)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        max_epochs=100,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        logger=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        accumulate_grad_batches=10,
        default_root_dir="outputs/",
        auto_scale_batch_size="binsearch",
        precision=16)
    # shell passes all values as strings
    masks = True if hparams.masks == "True" else False
    seq_length = int(hparams.seq_length)
    lr = float(hparams.lr)
    batch_size = int(hparams.batch_size)
    image_shape = (int(hparams.size), 2 * int(hparams.size))
    layers = int(hparams.layers)
    freeze = int(hparams.freeze)
    step = int(hparams.step)

    match hparams.backbone:
        # BASELINE MODEL : 1 input image, no pretraining
        case "baseline":
            channels = 3
            model = fcn_resnet50(pretrained=False,
                                 num_classes=2,
                                 pretrained_backbone=False)
            state_dict = torch.load("weights/fcn_resnet50_coco-1167a1af.pth")
            # Pretrained classifier expects 21 classes,
            # remove and ignore missing keys
            for key in list(state_dict.keys()):
                if "classifier" in key:
                    del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
            for child in model.backbone.children():
                for param in child.parameters():
                    param.requires_grad = False
        # RESNET_BACKBONE : 2 input images, no pre-training
        case "resnet":
            channels = 3
            model = ResNet_backbone(layers=layers,
                                    freeze=freeze,
                                    masks=masks)
        # ResUNet model
        case "resunet":
            channels = 1
            model = ResUNet(masks=masks,
                            freeze=freeze,
                            seq_length=seq_length)

        case "attention":
            channels = 1
            out_chan = 2 if masks else 1
            model = AttentionUNet(seq_length - 1, out_chan)
            if freeze > 0:
                c = 0
                for child in model.children():
                    for param in child.parameters():
                        param.requires_grad = False
                    if c == freeze:
                        break
                    c += 1

        case _:  # default cases
            print("model not specified. Exiting...")
            exit(1)

    if hparams.checkpoint:
        model = LightningModel.load_from_checkpoint(
            hparams.checkpoint,
            base_model=model,
            lr=lr,
            train_path=hparams.train_path,
            valid_path=hparams.valid_path,
            image_shape=image_shape,
            batch_size=batch_size,
            shuffle=True,
            masks=masks,
            seq_length=seq_length,
            step=step,
            channels=channels,
            strict=False)
    else:
        model = LightningModel(
            base_model=model,
            lr=lr,
            train_path=hparams.train_path,
            valid_path=hparams.valid_path,
            image_shape=image_shape,
            batch_size=batch_size,
            shuffle=True,
            masks=masks,
            seq_length=seq_length,
            step=step,
            channels=channels)

    if hparams.testing == "True":  # argparse makes everything strings
        trainer.test()
    else:
        trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--train_path", default="data")
    parser.add_argument("--valid_path", default="data/test")
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--batch_size", default=5)
    parser.add_argument("--masks", default=False)
    parser.add_argument("--checkpoint", default=False)
    parser.add_argument("--backbone", default="resunet")
    parser.add_argument("--step", default=1)
    parser.add_argument("--seq_length", default=2)
    parser.add_argument("--freeze", default=0)
    parser.add_argument("--size", default=512)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--layers", default=50)
    parser.add_argument("--testing", default=False)

    hparams = parser.parse_args()

    main(hparams)

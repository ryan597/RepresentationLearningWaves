import os
import argparse

import pytorch_lightning as pl
import torch
from torchvision.models.segmentation import fcn_resnet50

from backbones import ResNet_backbone, ResUNet, AttentionUNet
from lightningmodel import LightningModel, reset_all_weights


def main(hp, *args):
    if hp.masks:
        os.makedirs(f"../scratch/outputs/figures/training/masks/{os.environ['SLURM_JOB_ID']}", exist_ok=True)
        os.makedirs(f"../scratch/outputs/figures/validation/masks/{os.environ['SLURM_JOB_ID']}", exist_ok=True)
    else:
        os.makedirs(f"../scratch/outputs/figures/training/frames/{os.environ['SLURM_JOB_ID']}", exist_ok=True)
        os.makedirs(f"../scratch/outputs/figures/validation/frames/{os.environ['SLURM_JOB_ID']}", exist_ok=True)

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="train_loss_epoch",
                                                                    min_delta=0,
                                                                    mode="min",
                                                                    patience=10,
                                                                    strict=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss",
                                                       save_weights_only=True,
                                                       save_top_k=2,
                                                       filename="{epoch:02d}_{val_loss:.4f}")

    trainer = pl.Trainer(
        devices=hp.devices,
        accelerator=hp.accelerator,
        num_nodes=hp.num_nodes,
        max_epochs=100,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        logger=True,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        default_root_dir="../scratch/outputs/",
        precision="16-mixed",
        benchmark=True,
        profiler=hp.profiler,
        callbacks=[checkpoint_callback,
                   early_stop_callback])

    image_shape = (hp.size, 2 * hp.size)

    match hp.backbone:
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
            model = ResNet_backbone(layers=hp.layers,
                                    freeze=hp.freeze,
                                    masks=hp.masks)
        # ResUNet model
        case "resunet":
            channels = 1
            model = ResUNet(hp.seq_length - 1,
                            1 + hp.masks,
                            masks=hp.masks,
                            pretrain_bn=not hp.masks)

        case "attention":
            channels = 1
            model = AttentionUNet(hp.seq_length - 1,
                                  1 + hp.masks,
                                  masks=hp.masks,
                                  pretrain_bn=not hp.masks)

        case _:  # default cases
            print("model not specified. Exiting...")
            exit(1)

    if hp.checkpoint:
        model = LightningModel.load_from_checkpoint(
            hp.checkpoint,
            base_model=model,
            lr=hp.lr,
            train_path=hp.train_path,
            valid_path=hp.valid_path,
            image_shape=image_shape,
            batch_size=hp.batch_size,
            shuffle=True,
            masks=hp.masks,
            seq_length=hp.seq_length,
            step=hp.step,
            channels=channels,
            strict=False)
    else:
        model = LightningModel(
            base_model=model,
            lr=hp.lr,
            train_path=hp.train_path,
            valid_path=hp.valid_path,
            image_shape=image_shape,
            batch_size=hp.batch_size,
            shuffle=True,
            masks=hp.masks,
            seq_length=hp.seq_length,
            step=hp.step,
            channels=channels)

    if hp.backbone in ["resunet", "attention"]:
        if hp.masks:
            for i, child in enumerate(model.children()):
                if i > 5:  # reset the decoder if training segmentation
                    child.apply(fn=reset_all_weights)
        if hp.freeze > 0:
            for param in model.model.Conv1.parameters():
                param.requires_grad = False
        if hp.freeze > 1:
            for param in model.model.Conv2.parameters():
                param.requires_grad = False
        if hp.freeze > 2:
            for param in model.model.Conv3.parameters():
                param.requires_grad = False
        if hp.freeze > 3:
            for param in model.model.Conv4.parameters():
                param.requires_grad = False
        if hp.freeze > 4:
            for param in model.model.Conv5.parameters():
                param.requires_grad = False

    trainer.fit(model)

    trainer.test(model, ckpt_path="best")

    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data", type=str)
    parser.add_argument("--valid_path", default="data/test", type=str)
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument('--masks', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--backbone", default="resunet", type=str)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--seq_length", default=2, type=int)
    parser.add_argument("--freeze", default=0, type=int)
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--layers", default=50, type=int)
    parser.add_argument('--testing', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--devices", default=2, type=int)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--profiler", default=None, type=str)

    hparams = parser.parse_args()

    main(hparams)

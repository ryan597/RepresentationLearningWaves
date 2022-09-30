import torch
import pytorch_lightning as pl
from torchvision.models.segmentation import fcn_resnet50

from backbones import ResNet_backbone, ResUNet
from AttnUnet import AttentionUNet
from lightningmodel import LightningModel


def main(hparams, *args):
    pl.utilities.seed.seed_everything(2022)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        logger=True,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        accumulate_grad_batches=50,
        default_root_dir="outputs/",
        auto_scale_batch_size="binsearch")
    # shell passes all values as strings
    masks = True if hparams.masks == "True" else False
    seq_length = int(hparams.seq_length)
    lr = float(hparams.lr)
    batch_size = int(hparams.batch_size)
    image_shape = (int(hparams.size), 2 * int(hparams.size))
    layers = int(hparams.layers)
    freeze = int(hparams.freeze)

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
            model = AttentionUNet(seq_length-1, out_chan)

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
            channels=channels)
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
            channels=channels)

    trainer.fit(model)

    trainer.save_checkpoint("outputs/model_end.ckpt")

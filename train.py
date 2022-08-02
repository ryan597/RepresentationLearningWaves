import argparse

import torch
import pytorch_lightning as pl
from torchvision.models.segmentation import fcn_resnet50

from backbones import ResNet_backbone
from lightningmodel import LightningModel

# Disable Randomness
pl.utilities.seed.seed_everything(2022)


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
    parser.add_argument("--size",
                        help="Image shape as (SIZE, 2*SIZE)")
    parser.add_argument("--lr",
                        help="Initial learning rate",
                        default=0.001)
    parser.add_argument("--batch_size",
                        help="Number of samples to include in each batch")
    parser.add_argument("--backbone",
                        help="Backbone of model, resnet or resunet",
                        default="resnet")
    parser.add_argument("--layers",
                        help="How many layers of ResNet to use (18 or 50)",
                        default=50)
    parser.add_argument("--freeze",
                        help="How many layers of the backbone to freeze")
    parser.add_argument("--masks",
                        help="Train for segmentation or frame prediciton",
                        default=False)
    parser.add_argument("--dual",
                        help="Whether to use single or dual image inputs",
                        default=False)
    parser.add_argument("--checkpoint",
                        help="Path to checkpoint",
                        default=False)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)
    # shell passes all values as strings
    masks = True if args.masks == "True" else False
    dual = True if args.dual == "True" else False
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    image_shape = (int(args.size), 2*int(args.size))
    layers = int(args.layers)
    freeze = int(args.freeze)

    match args.backbone:
        # BASELINE MODEL : 1 input image, no pretraining
        case "baseline":
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
            model = ResNet_backbone(layers=layers,
                                    freeze=freeze,
                                    masks=masks,
                                    dual=dual)
        # ResUNet model

    if args.checkpoint:
        model = LightningModel.load_from_checkpoint(args.checkpoint,
                                                    base_model=model,
                                                    lr=lr,
                                                    train_path=args.train_path,
                                                    valid_path=args.valid_path,
                                                    image_shape=image_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    masks=masks,
                                                    dual=dual)

    else:
        model = LightningModel(base_model=model,
                               lr=lr,
                               train_path=args.train_path,
                               valid_path=args.valid_path,
                               image_shape=image_shape,
                               batch_size=batch_size,
                               shuffle=True,
                               masks=masks,
                               dual=dual)

    trainer.fit(model)
    trainer.save_checkpoint("outputs/model_end.ckpt")

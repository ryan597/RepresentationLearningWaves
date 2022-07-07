import argparse

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
    parser.add_argument("--lr",
                        help="Initial learning rate",
                        default=0.001)
    parser.add_argument("--backbone",
                        help="Backbone of model, resnet or resunet",
                        default="resnet")
    parser.add_argument("--batch_size",
                        help="Number of samples in each batch")
    parser.add_argument("--layers",
                        help="How many layers of ResNet to use (18 or 50)",
                        default=50)
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

    masks = True if args.masks == "True" else False
    dual = True if args.dual == "True" else False
    batch_size = int(args.batch_size)

    if args.backbone == "resnet":
        model = ResNet_backbone(layers=int(args.layers),
                                freeze=5,
                                dual=dual)
    # ResUNet model...

    # BASELINE
    model = fcn_resnet50(weights=None,
                         num_classes=2)

    if args.checkpoint:
        model = LightningModel.load_from_checkpoint(args.checkpoint,
                                                    base_model=model,
                                                    lr=float(args.lr),
                                                    train_path=args.train_path,
                                                    valid_path=args.valid_path,
                                                    image_shape=(512, 1024),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    masks=masks,
                                                    dual=dual)

    else:
        model = LightningModel(base_model=model,
                               lr=float(args.lr),
                               train_path=args.train_path,
                               valid_path=args.valid_path,
                               image_shape=(512, 1024),
                               batch_size=batch_size,
                               shuffle=True,
                               masks=masks,
                               dual=dual)

    trainer.fit(model)

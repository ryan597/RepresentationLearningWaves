import argparse

import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay

import data_utils
from backbones import ResNet_backbone, ResUNet, AttentionUNet
from lightningmodel import LightningModel


def get_auc_for_model(model, dataloader):
    pass


def main(args):
    shape = (args.size, args.size * 2)
    model = AttentionUNet(args.seq_length - 1, 2, True, False)
    model = LightningModel.load_from_checkpoint(checkpoint_path=args.checkpoint, base_model=model)
    model.to('cuda')
    model.eval()
    dataloader = data_utils.load_data(args.train_path, shape, args.batch_size,
                                      False, True, args.seq_length, args.step)

    y_true, y_pred = np.array(), np.array()
    for i, [inputs, labels] in enumerate(dataloader):
        inputs = inputs.to('cuda')
        outputs = model(inputs)
        np.append(y_pred, outputs.cpu()[:, 0].numpy().flatten())
        np.append(y_true, labels[:, 0].numpy().flatten())

    print(type(y_pred))
    print(type(y_pred[0]))
    roc = RocCurveDisplay.from_predictions(y_true, y_pred)
    roc.plot()
    plt.title(f"ROC curve for {args.backbone} model")
    plt.save_fig("../scratch/outputs/figures/roc/"
                    f"{args.backbone}_s{args.step}_l{args.seq_length}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--backbone")
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--size", type=int)

    args = parser.parse_args()

    with torch.no_grad():
        main(args)

import os
import argparse

import matplotlib.pyplot as plt
import torch

from lightningmodel import LightningModel
from backbones import AttentionUNet
import data_utils

def load_model(hp):
    model = AttentionUNet(hp.seq_length - 1,
                          hp.masks + 1,
                          masks=False,
                          pretrain_bn=False)

    model = LightningModel().load_from_checkpoint(checkpoint_path="hp.checkpoint", base_model=model)
    model.eval()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--seq_length")
    parser.add_argument("--output_path")

    batch_size = 10
    image_shape = (512, 1024)

    hp = parser.parse_args()

    model = load_model(hp)
    data = data_utils.load_data(hp.data_path, image_shape, batch_size, shuffle=False, seq_length=hp.seq_length)

    for i, batch in enumerate(data):
        input_batch, label_batch = batch
        input_batch = input_batch.cuda()
        label_batch = label_batch.cuda()


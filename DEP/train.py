"""
Main python script for the training of the dynamic texture model
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/RepresentationLearningWaves

###############################################################################

# Python imports
import os
import argparse
import json
import numpy as np
# import seaborn as sns
from os.path import exists

# Pytorch imports
import torch

# Other imports
import data_utils
from models.ResUNet import ResUNet
from models.methods import PyTorchModel

# Randomness must be disabled for distributed training!
SEED = 24
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


###############################################################################
# Parse arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Name of the config file inside ./config/")
    args = parser.parse_args()

    # config variables
    with open("configs/" + args.config + ".json", 'r') as config_json:
        config = json.load(config_json)
    
    print(json.dumps(config, indent=4), flush=True)

    model_name = config["model_name"]
    weights_path = config["weights_path"]
    train_path = config["train_path"]
    results_path = config["results_path"]
    valid_path = config['valid_path']
    image_width = config["image_width"]
    image_height = config["image_height"]
    imaug = config["imaug"]

    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    criterion = getattr(torch.nn, config["criterion"])
    optimizer = getattr(torch.optim, config["optimizer"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])

    # Loading datasets
    train_data = data_utils.load_data(train_path, (image_height, image_width), batch_size=10, shuffle=True)
    valid_data = data_utils.load_data(valid_path, (image_height, image_width), batch_size=10, shuffle=True)
    #data_utils.show_samples(train_data)

    # Loading model
    model = ResUNet(in_channels=2,
                    out_channels=1,
                    block_sizes=[32, 64, 128, 256, 512, 1024],
                    depths=[2, 3, 5, 3, 2])

    model = model.to("cuda:0")
    if exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cuda:0"))

    model = PyTorchModel(model,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         mode="min",
                         factor=0.1,
                         patience=2,
                         threshold=0.0001,
                         verbose=True)  # scheduler kwarg

    # Training model
    logs = model.train_model(train_data, valid=valid_data)

    model.save_model(model_name)
    model.save_logs(results_path)


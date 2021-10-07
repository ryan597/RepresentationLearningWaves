"""
Main python script for the training of the dynamic texture model
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/RepresentationLearningWaves

###############################################################################

# Python imports
import argparse
import json
import datetime
# import seaborn as sns
from os.path import exists

# Pytorch imports
import torch

# Other imports
import utils
from models.ResUNet import ResUNet
from models.methods import PyTorchModel

###############################################################################
# Parse arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Name of the config file inside ./config/")
    args = parser.parse_args()

    IMAGE_SIZE = 300

    # config variables
    with open("configs/" + args.config + ".json", 'r') as config_json:
        config = json.load(config_json)

    model_name = config["model_name"]
    weights_path = config["weights_path"]
    train_path = config["train_path"]
    results_path = config["results_path"]
    imaug = config["imaug"]

    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    criterion = config["criterion"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    # Loading datasets
    train_data = utils.load_data(train_path, IMAGE_SIZE)
    # valid_data = utils.load_data(valid_path, IMAGE_SIZE)

    # Loading model
    model = ResUNet(in_channels=2,
                    out_channels=1,
                    layer_size=[64, 128, 256, 512, 1024])

    if exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    model = PyTorchModel(model,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         T_max=epochs*3,  # scheduler kwarg
                         eta_min=1e-7)  # scheduler kwarg

    utils.show_samples(train_data)

    # Training model
    logs = model.train_model(train_data, valid_data=None)

    timestamp = datetime.datetime.today().replace(second=0, microsecond=0)
    with open(f"outputs/results/{timestamp}.json") as f:
        json.dump(logs, f)

    # Results

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
import data_utils
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
        print(config, flush=True)

    model_name = config["model_name"]
    weights_path = config["weights_path"]
    train_path = config["train_path"]
    results_path = config["results_path"]
    image_size = config["image_size"]
    imaug = config["imaug"]

    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    criterion = getattr(torch.nn, config["criterion"])
    optimizer = getattr(torch.optim, config["optimizer"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])

    # Loading datasets
    train_data = data_utils.load_data(train_path, (image_size, image_size))
    # valid_data = data_utils.load_data(valid_path, image_size)

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

    data_utils.show_samples(train_data)

    # Training model
    logs = model.train_model(train_data, valid_data=None)

    timestamp = datetime.datetime.today().replace(second=0, microsecond=0)
    with open(f"outputs/results/{timestamp}.json") as f:
        json.dump(logs, f)

    # Results

"""
Main python script for the training of the dynamic texture model
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/DynamicTextureWaves

###############################################################################

# Python imports
import argparse
import json
# import seaborn as sns
from os.path import exists

# Pytorch imports
import torch

# My imports
import utils
from models.ResUNet import ResUNet

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

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

    # Loading datasets
    train_data = utils.load_data(train_path, IMAGE_SIZE)

    # Loading model
    model = ResUNet(in_channels=2,
                    out_channels=1,
                    layer_size=[64, 128, 256, 512, 1024])

    model.to(DEVICE)

    if exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    # utils.show_samples(train_data)

    # Training model

    # Results

###############################################################################
# Save Models

# #torch.save(model.state_dict(), f"DT_model_spill_128.pth")
# torch.save(model.state_dict(), f"DT_model_spill_400_pretrained.pth")

# #torch.save(model.state_dict(), f"DT_model_plunge_128.pth")
# torch.save(model.state_dict(), f"DT_model_plunge_400_pretrained.pth")

# torch.save(model.state_dict(), f"DT_model_nonbreaking_128.pth")
# torch.save(model.state_dict(), f"DT_model_nonbreaking_400.pth")

###############################################################################
